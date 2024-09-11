from human_eval.data import write_jsonl, read_problems
import requests
from tqdm import tqdm

import time
import json

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

def generate_one_completion(prompt, templet):
    norm_model = 'http://129.254.177.83:5002'
    greedy_model = 'http://129.254.177.83:5001'
    
    res = []
    for s, c in templet:
        result = llama_submit(s.format(prompt), norm_model)
        sub_tasks = extract_class_text(result)
        if sub_tasks == -1:
            while sub_tasks == -1:
                result = llama_submit(s.format(prompt), norm_model)
                sub_tasks = extract_class_text(result)

        modified_code_result = llama_submit(c.format(prompt, sub_tasks), greedy_model)
        final_code = extract_class_code(modified_code_result)
        if final_code == -1:
            while final_code == -1:
                result = llama_submit(s.format(prompt), norm_model)
                sub_tasks = extract_class_text(result)
                modified_code_result = llama_submit(c.format(prompt, sub_tasks), greedy_model)
                final_code = extract_class_code(modified_code_result)
        
        res.append([sub_tasks, final_code])

    return res


from evalplus.data import get_mbpp_plus, write_jsonl

num_samples_per_task = 10

n_samples = [[] for _ in range(3)]
f_samples = [[] for _ in range(3)]

tmp = [[sub_prompt_so, code_templete_so], [sub_prompt_pl, code_templete_pl], [sub_prompt_ps, code_templete_ps]]
for task_id, prob in tqdm(get_mbpp_plus().items()):
    t1 = prob['canonical_solution'].split("\n")
    t = ""
    for line in t1:
        if prob['entry_point'] in line:
            t = t + "\n" + line
            break
        else:
            t = t + "\n" + line
    for _ in range(num_samples_per_task):
        result = generate_one_completion(prob['prompt'] + t, tmp)
        for n, d in enumerate(result):
            n_samples[n].append(dict(task_id=task_id, completion=d[0]))
            f_samples[n].append(dict(task_id=task_id, completion=d[1]))

write_jsonl("mbpp_OP_8_s.jsonl", n_samples[0])
write_jsonl("mbpp_OP_8.jsonl", f_samples[0])
write_jsonl("mbpp_PL_8_s.jsonl", n_samples[1])
write_jsonl("mbpp_PL_8.jsonl", f_samples[1])
write_jsonl("mbpp_PS_8_s.jsonl", n_samples[2])
write_jsonl("mbpp_PS_8.jsonl", f_samples[2])