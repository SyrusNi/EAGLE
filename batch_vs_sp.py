from eagle.modelbsne1.utils import *
from eagle.modelbsne1.ea_model import EaModel
from eagle.modelbsne1.kv_cache import initialize_past_key_values
from eagle.modelbsne1.choices import *

from fastchat.model import get_conversation_template
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
import time
from contextlib import contextmanager

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

import os
import json

answer_file = 'batch_vs_sp.json'
max_steps = 500
test_time = 10
bs = [1, 2, 4, 8, 12, 16, 20]
prefix = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Hi, could you share a tale about a charming llama that grows eagle-like hair and starts its own coffee shop? ASSISTANT:"

def normal_forward(model, tokenizer, prompt, max_steps):
    tokens = tokenizer(prompt, return_tensors="pt", padding=True).to("cuda")
    input_ids = tokens.input_ids
    attention_mask = tokens.attention_mask
    output = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens = max_steps)

    valid_tokens = output.ne(0)
    new_tokens = valid_tokens.sum(dim=-1) - attention_mask.sum(dim=-1)
    return sum(new_tokens)

def ea_forward(model, tokenizer, prompt, max_steps):
    input_s = tokenizer(prompt, return_tensors="pt", padding=True).to("cuda")
    outputs = model.eagenerate(input_s.input_ids,input_s.attention_mask, temperature=0.0, max_new_tokens=max_steps, top_k=15)
    init_len = input_s.input_ids.shape[1]
    new_tokens = 0
    for output in outputs:
        new_tokens += len(output) - init_len
    return new_tokens

def test_throughout(func, model, tokenizer, prompt):
    new_tokens = 0
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(test_time):
        new_tokens += func(model, tokenizer, prompt, max_steps)
    torch.cuda.synchronize()
    end = time.time()
    throughout = new_tokens / (end-start)
    return throughout

def main():
    base_model_path = 'models/vicuna-7b-v1.3'
    ea_model_path = 'models/EAGLE-Vicuna-7B-v1.3'

    # base part
    base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            # load_in_8bit=True,
            device_map="auto"       
    )
    base_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, padding_side = 'left')

    throughout_list = []
    for b in bs:
        prompt = [prefix] * b
        throughout = test_throughout(normal_forward, base_model, tokenizer, prompt)
        throughout_list.append(float(throughout.cpu()))
    
    '''
    #os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    with open(os.path.expanduser(answer_file), "a") as fout:
        record = {'batch_size': list(bs), 'throughout': throughout_list}
        fout.write(json.dumps(record) + "\n")
    '''
    # ea part
    ea_model = EaModel.from_pretrained(
            base_model_path,
            ea_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            # load_in_8bit=True,
            device_map="auto"       
    )

    ea_model.eval()
    ea_model.tokenizer.padding_side = "left"
    ea_model.tokenizer.pad_token = ea_model.tokenizer.eos_token
    ea_model.config.pad_token_id = ea_model.config.eos_token_id

    throughout_list_2 = []
    for b in bs:
        prompt = [prefix] * b
        throughout = test_throughout(ea_forward, ea_model, ea_model.tokenizer, prompt)
        throughout_list_2.append(throughout)

    print(bs, throughout_list, throughout_list_2)
    # dump record
    with open(os.path.expanduser(answer_file), "w") as fout:
        record = {'batch_size': list(bs), 'throughout': throughout_list, 'eagle_throughout':throughout_list_2}
        fout.write(json.dumps(record))

if __name__ == '__main__':
    main()