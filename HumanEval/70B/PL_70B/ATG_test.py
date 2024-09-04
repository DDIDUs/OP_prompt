from human_eval.data import write_jsonl, read_problems
import requests
from tqdm import tqdm
import pickle

import time
import json


plan_prompt = '''
Problem:
```python
{}
```

Write a plan for the problem. 
DO NOT generate any test cases or code
Package your response in ```plaintext ... ```.
'''

codegen_prompt = '''
Problem:
```
{}
```
Plan:
```
{}
```

Analyse the problem, refer to the plan and write a Python code to solve the problem.
DO NOT generate any test cases or descriptions. 
Package your code in ```python ... ```.
'''

fix_prompt = '''
generated code:
```
{}
```

problem:
```
{}
```
Analyze the provided code to understand its structure and logic.
Identify potential functional errors and suggest appropriate fixes to mitigate them.
If the code is already functioning correctly, no changes will be recommended.
DO NOT generate any test cases or descriptions or comment. 
Package your code in ```python ... ```.
'''

def extract(result, s):
    tmp = result[result.find(s) + len(s):]
    return tmp[:tmp.find("```")]

def extract_class_code(result):
    if "```python" in result:
        return extract(result, "```python")
    elif "```\npython" in result:
        return extract(result, "```\npython")
    elif "```\n" in result:
        return extract(result, "```\n")
    else:
        return -1
    
def extract_class_text(result):
    if "```plaintext" in result:
        return extract(result, "```plaintext")
    elif "```\nplaintext" in result:
        return extract(result, "```\nplaintext")
    elif "```\n" in result:
        return extract(result, "```\n")
    else:
        return -1
    
def llama_submit(prompt, url):
    url = "{}/v1/generateText".format(url)

    headers = {"Content-Type": "application/json"}
    data = {"prompt": "{}".format(prompt)}

    start_time = time.time()
    response = requests.post(url, headers=headers, data=json.dumps(data))
    end_time = time.time()
    latency = end_time - start_time
    result = json.loads(response.text)['text']
    #print(f"Latency: {latency} seconds")
    #print("LLM response: " + result)
    
    return result

def llama_3_submit(instruction, url):
    
    server_url = url
    
    input_data = {
        'input': '{}'.format(instruction)
    }

    # 서버에 POST 요청 보내기
    response = requests.post(server_url, json=input_data)

    # 서버로부터 응답 받기
    if response.status_code == 200:
        result = response.json()
        re = result['response']
    else:
        print("Failed to get response from server, status code:", response.status_code)
        
    return re

def generate_one_completion(prompt):
    
    norm_model = 'http://129.254.177.85:5000'
    greedy_model = 'http://129.254.177.85:5001'
    
    result = llama_submit(plan_prompt.format(prompt), norm_model)
    plan = extract_class_text(result)
    if plan == -1:
        while plan == -1:
            result = llama_submit(plan_prompt.format(prompt), norm_model)
            plan = extract_class_text(result)

    result = llama_submit(codegen_prompt.format(prompt, plan), greedy_model)
    pre_code = extract_class_code(result)
    if pre_code == -1:
        while pre_code == -1:
            result = llama_submit(plan_prompt.format(prompt), norm_model)
            plan = extract_class_text(result)
            result = llama_submit(codegen_prompt.format(prompt, plan), greedy_model)
            pre_code = extract_class_code(result)

    modified_code_result = llama_submit(fix_prompt.format(pre_code, prompt), greedy_model)
    final_code = extract_class_code(modified_code_result)
    if final_code == -1:
        while final_code == -1:
            result = llama_submit(plan_prompt.format(prompt), norm_model)
            plan = extract_class_text(result)
            result = llama_submit(codegen_prompt.format(prompt, plan), greedy_model)
            pre_code = extract_class_code(result)
            modified_code_result = llama_submit(fix_prompt.format(pre_code, prompt), greedy_model)
            final_code = extract_class_code(modified_code_result)
            
    return pre_code, final_code
    
problems = read_problems()

num_samples_per_task = 10

n_samples = []
f_samples = []

for task_id in tqdm(problems):
    for _ in range(num_samples_per_task):
        n_code, f_code = generate_one_completion(problems[task_id]["prompt"])
        n_samples.append(dict(task_id=task_id, completion=n_code))
        f_samples.append(dict(task_id=task_id, completion=f_code))

write_jsonl("samples_ATG_n_2.jsonl", n_samples)
write_jsonl("samples_ATG_2.jsonl", f_samples)