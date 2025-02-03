import torch
from torch import topk
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

Tiny_llama_1B = 'models/vicuna-7b-v1.3'

tokenizer = AutoTokenizer.from_pretrained(Tiny_llama_1B, padding_side = 'left') # 默认填充方向是右侧，在推理场景下不好用
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token # Most LLMs don't have a pad token by default
model = AutoModelForCausalLM.from_pretrained(Tiny_llama_1B, device_map='cuda:0', torch_dtype=torch.float32) # 正常情况一张卡是放不下的

from typing import List, Optional, Tuple, Union

def beam_search(model, input_ids, attention_mask, depth, beam_nums):
    '''
    beam_search: keep top beam_nums tokens in each depth
    
    Inputs:
    - input_ids: [b, s]
    - attention_mask: [b, s]
    '''
    bs = input_ids.shape[0] # =1 here
    seq_len = attention_mask.sum(dim=-1)[:, None]
    
    outputs = model(input_ids, attention_mask=attention_mask)
    logits, past_key_values = outputs.logits, outputs.past_key_values

    # [bs, v]
    log_prob = F.log_softmax(logits[:, -1], dim=-1)
    # [bs, topk]
    topk_tokens = topk(log_prob, beam_nums)
    topk_p, topk_index = topk_tokens.values, topk_tokens.indices
    scores = topk_p

    # convert 2d mask to 3d mask
    #attention_mask = _expand_mask(attention_mask, torch.float32, beam_nums)
    attention_mask = attention_mask[:, None].repeat(1, beam_nums, 1)
    attention_mask_init = torch.diag(torch.ones(beam_nums))[None].repeat(bs, 1, 1).to(attention_mask.device)

    for i in range(depth):
        attention_mask = torch.cat([attention_mask, attention_mask_init], dim=-1)
        #tree_attention = _prepare_decoder_attention_mask(attention_mask, (bs, input_ids.shape[1]), input_ids, past_key_values[0][0].shape[2])
        position_ids = torch.ones(bs, beam_nums).to(model.device) + seq_len.repeat(1, beam_nums)
        outputs = model(topk_index, 
                        position_ids = position_ids, 
                        past_key_values = past_key_values, 
                        attention_mask = attention_mask[:, None], 
                        use_cache = True)
        # [bs, topk, v]
        logits, past_key_values = outputs.logits, outputs.past_key_values
        log_prob = F.log_softmax(logits.view(bs*beam_nums, -1), dim=-1)
        # [bs*topk, topk]
        leaf_tokens = topk(log_prob, beam_nums)
        topk_p_leaf, topk_index_leaf = leaf_tokens.values, leaf_tokens.indices
        
        # [bs, topk, topk]
        scores = topk_p_leaf.view(bs, beam_nums, -1) + scores[:, :, None].repeat(1, 1, beam_nums)
        # [bs, topk]
        topk_tokens = topk(scores.view(bs, -1), beam_nums)
        topk_p, topk_index = topk_tokens.values, topk_tokens.indices

        parent_nodes = topk_index // beam_nums
        b = tuple(torch.arange(bs))
        b = torch.arange(bs)[:, None]
        # inheret mask info from parent nodes
        attention_mask = attention_mask[b, parent_nodes]

        scores = topk_p
        # [bs, topk]
        topk_index = topk_index_leaf.view(bs, -1).gather(dim=1, index=topk_index)
        seq_len += 1
    

tokens = tokenizer(['Hello', 'quantum computation'], return_tensors='pt', padding=True).to(model.device)
generate_ids = beam_search(model, tokens.input_ids, tokens.attention_mask, 5, 2)
tokenizer.batch_decode(generate_ids, skip_special_tokens=True)