from human_eval.data import write_jsonl, read_problems
import requests
from tqdm import tqdm

code_templete = '''
Problem:
```python
{}
```

write the python code for the problem.
DO NOT generate any test cases or descriptions. Package your code in ```python ... ```.
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

def llama_3_submit(instruction):
    
    server_url = 'http://129.254.177.83:5002/predict'
    
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
    
    modified_code_result = llama_3_submit(code_templete.format(prompt))
    final_code = extract_class_code(modified_code_result)
    
    if final_code == -1:
        while final_code == -1:
            modified_code_result = llama_3_submit(code_templete.format(prompt))
            final_code = extract_class_code(modified_code_result)

    return final_code
    
problems = read_problems()

num_samples_per_task = 10

samples = [
    dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
    for task_id in tqdm(problems)
    for _ in range(num_samples_per_task)
]

write_jsonl("samples_norm.jsonl", samples)