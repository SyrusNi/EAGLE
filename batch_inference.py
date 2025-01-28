from eagle.modelbsne1.ea_model import EaModel
from fastchat.model import get_conversation_template
import torch

base_model_path = 'models/vicuna-7b-v1.3'
EAGLE_model_path = 'models/EAGLE-Vicuna-7B-v1.3'
#EAGLE_model_path = 'test_eagle_vicuna-7b-v1.3'

model = EaModel.from_pretrained(
    base_model_path=base_model_path,
    ea_model_path=EAGLE_model_path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto"
)
# left padding
model.eval()
model.tokenizer.padding_side = "left"
model.tokenizer.pad_token = model.tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

your_message="Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions."
conv = get_conversation_template("vicuna")
conv.append_message(conv.roles[0], your_message)
conv.append_message(conv.roles[1], None)
prompt1 = conv.get_prompt()+" "

your_message="Hello"
conv = get_conversation_template("vicuna")
conv.append_message(conv.roles[0], your_message)
conv.append_message(conv.roles[1], None)
prompt2 = conv.get_prompt()+" "

input_s=model.tokenizer([prompt1,prompt2],return_tensors="pt",padding=True).to("cuda")
output_ids=model.eagenerate(input_s.input_ids,input_s.attention_mask,temperature=0.0,max_new_tokens=512,top_k=15)
outputs=model.tokenizer.batch_decode(output_ids)

for output in outputs:
    print(output)

# vanilla auto-regression
# output_ids, new_token, idx=model.naivegenerate(input_s.input_ids,input_s.attention_mask,temperature=0.0,max_new_tokens=512,top_k=15,log=True)