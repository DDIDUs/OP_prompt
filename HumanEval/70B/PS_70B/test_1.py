from human_eval.data import write_jsonl, read_problems
import requests
from tqdm import tqdm

import time
import json

code_templete_gt = '''
{}

write a python code to solve the problem.
DO NOT generate any test cases or descriptions. Package your code in ```python ... ```.
'''

code_templete_so = '''
{}

Objectives:
{}

DO NOT generate any test cases or descriptions. Package your code in ```python ... ```.
'''

sub_prompt_so = '''
{{specific objectives}}
Write the independent objectives for solving the given problem concisely.

{}

Package your response in ```plaintext ... ```. DO NOT generate any code or description.
'''

code_templete_ps = '''
{}

Pseudo code:
```
{}
```
write a python code to solve the problem.
DO NOT generate any test cases or descriptions. Package your code in ```python ... ```.
'''

sub_prompt_ps = '''
{{specific pseudo code}}
Write the pseudo code for solving the given problem.

{}

Package your response in ```plaintext ... ```. DO NOT generate any code or description.
'''

code_templete_pl = '''
{}

plan:
{}

DO NOT generate any test cases or descriptions. 
Package your code in ```python ... ```.
'''

sub_prompt_pl = '''
{}

Write a plan for the problem. 
DO NOT generate any test cases or code. Package your response in ```plaintext ... ```.
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

def generate_one_completion(prompt, gt_prompt):
    norm_model = 'http://129.254.177.85:5000'
    greedy_model = 'http://129.254.177.85:5001'
    
    modified_code_result = llama_submit(code_templete_gt.format(gt_prompt), norm_model)
    gt_code = extract_class_code(modified_code_result)
    if gt_code == -1:
        while final_code == -1:
            modified_code_result = llama_submit(code_templete_gt.format(gt_prompt), norm_model)
            gt_code = extract_class_code(modified_code_result)
        
    result = llama_submit(sub_prompt_ps.format(prompt), norm_model)
    sub_tasks = extract_class_text(result)
    if sub_tasks == -1:
        while sub_tasks == -1:
            result = llama_submit(sub_prompt_ps.format(prompt), norm_model)
            sub_tasks = extract_class_text(result)
            
    modified_code_result = llama_submit(code_templete_ps.format(prompt, sub_tasks), greedy_model)
    final_code = extract_class_code(modified_code_result)
    if final_code == -1:
        while final_code == -1:
            result = llama_submit(sub_prompt_ps.format(prompt), norm_model)
            sub_tasks = extract_class_text(result)
            modified_code_result = llama_submit(code_templete_ps.format(prompt, sub_tasks), greedy_model)
            final_code = extract_class_code(modified_code_result)

    return gt_code, sub_tasks, final_code


from evalplus.data import get_human_eval_plus, write_jsonl

num_samples_per_task = 10

n_samples = []
f_samples = []
gt_samples = []

for task_id, prob in tqdm(get_human_eval_plus().items()):
    for _ in range(num_samples_per_task):
        gt, st, fc = generate_one_completion(prob['prompt'], prob['prompt'] + prob['canonical_solution'])
        gt_samples.append(dict(task_id=task_id, completion=gt))
        n_samples.append(dict(task_id=task_id, completion=st))
        f_samples.append(dict(task_id=task_id, completion=fc))

write_jsonl("humaneval_GT_70.jsonl", gt_samples)
write_jsonl("humaneval_PS_70_s.jsonl", n_samples)
write_jsonl("humaneval_PS_70.jsonl", f_samples)