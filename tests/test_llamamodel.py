from srs.LLM.config import Config
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from model.Llama_manual_architecture import LlamaModel
from model.Llama_full_model import LlamaManager

import model.Llama_manual_architecture as LA
import importlib
importlib.reload(LA)
from model.Llama_manual_architecture import LlamaModel

import srs.inference.help as help
importlib.reload(help)

from srs.inference.help import (get_device, generate, _as_prompt, get_model)
from srs.training.help_func import (get_tokenizer, get_dataloader)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
import dataset.get_prep_data as ds_prep
import dataset.New_dataloader as DL

DATA, LANGUAGE, MAX_RANK = "all", "en", 1

train_df, val_df = ds_prep.get_data_preprocessed(Config, split_ratio=0.9, data=DATA,
                                                 language=LANGUAGE, max_rank=MAX_RANK)

MAX_LENGTH = 512             # tokens; longer conversations are cut after their last complete reply
BATCH_SIZE = 4               # lower to 2 if the Mac runs out of memory, then raise GRAD_ACCUM to 16
GRAD_ACCUM = 8               # effective batch = BATCH_SIZE * GRAD_ACCUM
SEED = 42
train_dl = DL.create_dataloader(train_df["text"].tolist(), tokenizer=get_tokenizer(),
                                batch_size=BATCH_SIZE, shuffle=True,
                                max_length=MAX_LENGTH, seed=SEED)

batch = next(iter(train_dl))
model = LlamaManager(use_lora="LoRA")
model = model.get_model(tokenizer=get_tokenizer())
model.train()
logits = model(batch["input_ids"], attention_mask=batch["attention_mask"])
loss = torch.nn.functional.cross_entropy(
    logits[:, :-1].reshape(-1, logits.size(-1)),
    batch["labels"][:, 1:].reshape(-1), ignore_index=-100)
loss.backward()
g = model.tok_emb.weight.grad
print(g.abs().sum(dim=1).nonzero().flatten().tolist())          # [1, 2]  — only the chat rows
print(model.trf_blocks[0].self_attn.q_proj.linear.weight.grad)  # None   — base frozen
print(model.trf_blocks[0].self_attn.q_proj.B.grad.abs().sum() > 0)   # True
print(model.trf_blocks[0].self_attn.q_proj.A.grad.abs().sum())       # 0 !

print(sum(p.numel() for p in model.parameters()))     # 134,515,008
print(model.out_head.weight is model.tok_emb.weight)  # True
print(generate(model, tok, "cpu", "What is the capital of France?", temperature=0, max_new_tokens=20))

from model.Llama_manual_architecture import LlamaModel, GroupedQueryAttention, FeedForward   # line 7
...
model = LlamaManager(use_lora="LoRA").get_model(tokenizer=tok)
model.train()

batch = next(iter(dl))          # a dataloader from val_all_en_r1.csv
logits = model(batch["input_ids"], attention_mask=batch["attention_mask"])
loss = torch.nn.functional.cross_entropy(
    logits[:, :-1].reshape(-1, logits.size(-1)),
    batch["labels"][:, 1:].reshape(-1), ignore_index=-100)
loss.backward()

g = model.tok_emb.weight.grad
print(g.abs().sum(dim=1).nonzero().flatten().tolist())          # [1, 2]  — only the chat rows
print(model.trf_blocks[0].self_attn.q_proj.linear.weight.grad)  # None   — base frozen
print(model.trf_blocks[0].self_attn.q_proj.B.grad.abs().sum() > 0)   # True
print(model.trf_blocks[0].self_attn.q_proj.A.grad.abs().sum())       # 0 !








emb_weights = model.tok_emb.weight
print(model.tok_emb.weight.shape[0])

print(repr(_as_prompt("What is the capital of France?")))

print(tok.encode("<|im_start|>user\n"))
print(tok.encode("<|im_end|>\n"))
print(tok.encode("\n"))
print(tok.eos_token, tok.eos_token_id)

hf_model = AutoModelForCausalLM.from_pretrained(REPO, torch_dtype=torch.float32).eval()


ids = tok("The capital of France is", return_tensors="pt").input_ids
with torch.no_grad():
    mine = model(ids)
    ref  = hf_model(ids).logits

print((mine - ref).abs().max())                     # expect ~1e-4 or smaller
print(tok.decode(mine[0, -1].argmax()))

with torch.no_grad():
    # prefill
    logits, caches = model(ids, use_cache=True)
    seq = ids
    for step in range(10):
        next_id = logits[:, -1].argmax(-1, keepdim=True)          # (1, 1)
        seq = torch.cat([seq, next_id], dim=1)

        # cached: only the new token
        logits, caches = model(next_id, kv_caches=caches,
                               start_pos=seq.shape[1] - 1, use_cache=True)
        # reference: full recompute
        full = model(seq)
        print(step, (logits[:, -1] - full[:, -1]).abs().max().item())   # each ~1e-4 or less

out = generate(model, tok, "cpu", "The capital of France is",
               max_new_tokens=10, temperature=0, repetition_penalty=1.0)
print(out)