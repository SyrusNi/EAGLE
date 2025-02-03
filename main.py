import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "3" # define GPU id, remove if you want to use all GPUs available
import torch
from tqdm import tqdm
import time
from contextlib import contextmanager
import numpy as np
from eagle.model.modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from eagle.model.ea_model import EaModel
from eagle.model.kv_cache import *
from eagle.model.utils import *
from eagle.model.choices import *
import transformers
from huggingface_hub import hf_hub_download

@contextmanager
def timed(wall_times, key):
    start = time.time()
    torch.cuda.synchronize()
    yield
    torch.cuda.synchronize()
    end = time.time()
    elapsed_time = end - start
    wall_times[key].append(elapsed_time)

@torch.no_grad()
def eagenerate(
        self,
        input_ids,
        temperature=0.0,
        top_p=0.0,
        top_k=0.0,
        max_new_tokens=512,
        max_length=2048,
        tree_choices=mc_sim_7b_63,

):
    wall_times = {'eagle': [], 'tree': [], 'posterior': [], 'update': [], 'init': []}

    with timed(wall_times, 'init'):
        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        #assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        if hasattr(self, "tree_choices") and self.tree_choices == tree_choices:
            tree_buffers = self.tree_buffers
        else:
            tree_buffers = generate_tree_buffers(
                tree_choices, device=self.base_model.model.layers[-1].self_attn.q_proj.weight.device
            )
            tree_buffers["retrieve_indices_head"] = tree_buffers["retrieve_indices"].to(
                self.base_model.lm_head.weight.device)
        self.tree_buffers = tree_buffers
        self.tree_choices = tree_choices

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        tree_logits, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, tree_buffers["tree_attn_mask"], past_key_values, logits_processor
        )
    
    with timed(wall_times, 'eagle'):
        new_token = 0

    for idx in range(max_length):
        with timed(wall_times, 'eagle'):
            candidates, cart_candidates_prob, tree_candidates = generate_candidates(
                tree_logits,
                tree_buffers["tree_indices"],
                tree_buffers["retrieve_indices"],
                sample_token,
                logits_processor
            )
        
        with timed(wall_times, 'tree'):
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                tree_candidates,
                past_key_values,
                tree_buffers["tree_position_ids"],
                input_ids,
                tree_buffers["retrieve_indices_head"],
            )
        
        with timed(wall_times, 'posterior'):
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor, cart_candidates_prob, tree_logits[2], tree_buffers["p_indices"],
                tree_candidates, tree_buffers["b_indices"]
            )

        with timed(wall_times, 'update'):
            input_ids, tree_logits, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                tree_buffers["retrieve_indices"],
                logits_processor,
                logits,
                tree_logits,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state,
                hidden_state_new,
                sample_p
            )

        if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            return input_ids, new_token, idx, wall_times
        if new_token > max_new_tokens:
            return input_ids, new_token, idx, wall_times
        if input_ids.shape[1] > max_length:
            return input_ids, new_token, idx, wall_times

@torch.no_grad()
def eagenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=True,
            is_llama3=False,

    ):
        wall_times = {'eagle': [], 'tree': [], 'posterior': [], 'update': [], 'init': []}

        with timed(wall_times, 'init'):
            if is_llama3:
                stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            max_length=max_length-self.ea_layer.total_tokens-10

            if temperature > 1e-5:
                logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
            else:
                logits_processor = None
            #assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
            # Avoid modifying the input_ids in-place

            padding=(torch.zeros(1,1,dtype=torch.long)-1).to(input_ids.device)
            input_ids = input_ids.clone()
            self.ea_layer.reset_kv()

            # Initialize the past key and value states
            if hasattr(self, "past_key_values"):
                past_key_values = self.past_key_values
                past_key_values_data = self.past_key_values_data
                current_length_data = self.current_length_data
                # Reset the past key and value states
                current_length_data.zero_()
            else:
                (
                    past_key_values,
                    past_key_values_data,
                    current_length_data,
                ) = initialize_past_key_values(self.base_model)
                self.past_key_values = past_key_values
                self.past_key_values_data = past_key_values_data
                self.current_length_data = current_length_data

            input_len = input_ids.shape[1]
            reset_tree_mode(self)
            draft_tokens, retrieve_indices,tree_mask,tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
                input_ids, self, past_key_values, logits_processor
            )
        new_token = 0

        for idx in range(max_length):
            #with Timer("all"):
            self.base_model.model.tree_mask = tree_mask

            draft_tokens=draft_tokens.to(input_ids.device)
            #with Timer("tree_decoding"):
            with timed(wall_times, 'tree'):
                logits, hidden_state_new, outputs = tree_decoding(
                    self,
                    draft_tokens,
                    past_key_values,
                    tree_position_ids,
                    input_ids,
                    retrieve_indices,
                )
            #retrieve_indices=tree_buffers["retrieve_indices"]
            #logits = logits[0, retrieve_indices]
            with timed(wall_times, 'posterior'):
                draft_tokens=torch.cat((draft_tokens,padding),dim=1)
                candidates=draft_tokens[0,retrieve_indices]

                best_candidate, accept_length, sample_p = evaluate_posterior(
                    logits, candidates, logits_processor
                )
            # print(accept_length)
            #with Timer("update_inference_inputs"):
            with timed(wall_times, 'update'):
                input_ids, draft_tokens, retrieve_indices,tree_mask,tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
                    input_ids,
                    candidates,
                    best_candidate,
                    accept_length,
                    retrieve_indices,
                    logits_processor,
                    new_token,
                    past_key_values_data,
                    current_length_data,
                    self,
                    hidden_state_new,
                    sample_p
                )

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx, wall_times
            

def main(base_model_path, EAGLE_model_path):

    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=EAGLE_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        #total_token=-1
    )
    tokenizer = model.get_tokenizer()

    model.eval()

    temperature = 0.
    posterior_threshold = 0.09
    posterior_alpha = 0.3

    prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Hi, could you share a tale about a charming llama that grows eagle-like hair and starts its own coffee shop? ASSISTANT:"

    with torch.inference_mode():
        input_ids = tokenizer([prompt]).input_ids
        output_ids, new_token, idx, wall_time = eagenerate(
                        model,
                        torch.as_tensor(input_ids).cuda(),
                        temperature
                    )
        output_ids = output_ids[0][len(input_ids[0]) :]
        print("Output length:", output_ids.size(-1))
        print("Compression ratio:", new_token / idx)

    output = tokenizer.decode(
                        output_ids,
                        spaces_between_special_tokens=False,
                    )
    print(output)

    max_length = 50

    def format_string(text, value, max_length):
        value_str = "{:.3f}".format(value)
        return f"{text:<{max_length - len(value_str)}}{value_str}"

    time_init = np.sum(wall_time['init'] )
    time_eagle = np.sum(wall_time['eagle'] )
    time_tree = np.sum(wall_time['tree'] )
    time_posterior = np.sum(wall_time['posterior'] )
    time_update = np.sum(wall_time['update'] )
    time_total = time_init + time_eagle + time_tree + time_posterior + time_update

    print('='*max_length)
    print(format_string("Wall time init: ", time_init, max_length))
    print(format_string("Wall time eagle: ", time_eagle, max_length))
    print(format_string("Wall time Tree: ", time_tree, max_length))
    print(format_string("Wall time Posterior: ", time_posterior, max_length))
    print(format_string("Wall time Update: ", time_update, max_length))
    print('-'*max_length)
    print(format_string("Wall time portion eagle: ", time_eagle / time_total, max_length))
    print(format_string("Wall time portion Tree: ", time_tree / time_total, max_length))
    print(format_string("Wall time portion Posterior: ", time_posterior / time_total, max_length))
    print(format_string("Wall time portion Update: ", time_update / time_total, max_length))
    print('-'*max_length)
    print(format_string("Tokens/second: ", new_token / time_total, max_length))
    print('='*max_length)

if __name__ == '__main__':
    base_model_path = 'models/vicuna-7b-v1.3'
    EAGLE_model_path = 'models/EAGLE-Vicuna-7B-v1.3'
    EAGLE_model_path = 'test_2_eagle_vicuna-7b-v1.3'
    main(base_model_path, EAGLE_model_path)


