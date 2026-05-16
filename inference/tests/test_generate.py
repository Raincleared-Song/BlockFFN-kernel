import torch
from llamacu.llama import LLM
from transformers import AutoTokenizer, AutoConfig
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default="SparseLLM/BlockFFN-v2-4.5B-A1.xB")
parser.add_argument("--dtype", type=str, default="bf16")
parser.add_argument("--disable-cuda-graph", action="store_true")
parser.add_argument("--use-kernel", action="store_true")
parser.add_argument("--num-generate", type=int, default=100)
args = parser.parse_args()

dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
path = args.model_path
cuda_graph = not args.disable_cuda_graph
use_kernel = args.use_kernel
num_generate = args.num_generate

prompt = "Beijing is the"
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
config = AutoConfig.from_pretrained(path, trust_remote_code=True)
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda().int()
num_tokens = input_ids.numel()

print(f"Prompt: {prompt}")
print(f"Input IDs: {input_ids}")
print("-" * 50)

position_ids = torch.arange(num_tokens, dtype=torch.int32, device="cuda").view(1, num_tokens)

llm = LLM(path, dtype=dtype, memory_limit=0.5, use_kernel=use_kernel, cuda_graph=cuda_graph)
our_generate = lambda: llm.generate(input_ids, num_generate)

llm.init_storage()
llm.load_from_hf()

tokens, _ = our_generate()
print("Generated tokens:", tokens)
print("Generated text:", tokenizer.decode(tokens))
