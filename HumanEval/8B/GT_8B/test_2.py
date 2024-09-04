from human_eval.data import write_jsonl, read_problems
import requests
from tqdm import tqdm

import time
import json

code_templete = '''
{}

write a python code to solve the problem.
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
    norm_model = 'http://129.254.177.83:5002'

    modified_code_result = llama_submit(code_templete.format(prompt), norm_model)
    final_code = extract_class_code(modified_code_result)
    if final_code == -1:
        while final_code == -1:
            modified_code_result = llama_submit(code_templete.format(prompt), norm_model)
            final_code = extract_class_code(modified_code_result)

    return final_code

    
problems = read_problems()

num_samples_per_task = 10

n_samples = []
#f_samples = []

for task_id in tqdm(problems):
    for _ in range(num_samples_per_task):
        n_code = generate_one_completion(problems[task_id]["prompt"] + problems[task_id]['canonical_solution'])
        n_samples.append(dict(task_id=task_id, completion=n_code))
        
        #f_samples.append(dict(task_id=task_id, completion=f_code))

write_jsonl("samples_GT_8B.jsonl", n_samples)