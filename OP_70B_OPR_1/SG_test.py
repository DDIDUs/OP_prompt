from human_eval.data import write_jsonl, read_problems
import requests
from tqdm import tqdm

import time
import json

code_templete = '''
{}

Problem:
```
{}
```
write a python code to solve the problem.
DO NOT generate any test cases or descriptions. Package your code in ```python ... ```.
'''

sub_prompt = '''
{{specific objectives}}
Present the specific objectives that need to be achieved in order to analyze and solve the given problem.

{{Priorities and relationships}}
Define the priorities for the specified objectives and detail the relationships between them.

{}

Package your response in ```plaintext ... ```. DO NOT generate any code or description.
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

    response = requests.post(url, headers=headers, data=json.dumps(data))
    result = json.loads(response.text)['text']
    
    return result

def generate_one_completion(prompt):
    norm_model = 'http://129.254.177.85:5000'
    greedy_model = 'http://129.254.177.85:5001'
    
    result = llama_submit(sub_prompt.format(prompt), norm_model)
    sub_tasks = extract_class_text(result)
    if sub_tasks == -1:
        while sub_tasks == -1:
            result = llama_submit(sub_prompt.format(prompt), norm_model)
            sub_tasks = extract_class_text(result)

    modified_code_result = llama_submit(code_templete.format(prompt, sub_tasks), greedy_model)
    final_code = extract_class_code(modified_code_result)
    if final_code == -1:
        while final_code == -1:
            modified_code_result = llama_submit(code_templete.format(prompt, sub_tasks), greedy_model)
            final_code = extract_class_code(modified_code_result)

    return final_code

    
problems = read_problems()

num_samples_per_task = 10

samples = [
    dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
    for task_id in tqdm(problems)
    for _ in range(num_samples_per_task)
]

write_jsonl("samples_SG_1.jsonl", samples)