from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from model.Llama_manual_architecture import LlamaModel
from srs.LLM.config import Config

model = LlamaModel(Config.LLAMA_CONFIG)

REPO = "HuggingFaceTB/SmolLM2-135M"

cfg_path = hf_hub_download(REPO, "config.json")
w_path   = hf_hub_download(REPO, "model.safetensors")
hf_sd = load_file(w_path)

print(len(hf_sd))
for k, v in list(hf_sd.items())[:15]:
    print(k, tuple(v.shape), v.dtype)

def rename_key(k):
    # 1. strip the "model." prefix
    if k.startswith("model."):
        k = k[len("model."):]

    # 2. "embed_tokens" → "tok_emb"
    k = k.replace("embed_tokens", "tok_emb")

    # 3. "layers." → "trf_blocks."
    if k.startswith("layers."):
        k = "trf_blocks." + k[len("layers."):]

    # 4. top-level "norm.weight" → "final_norm.weight"
    if k == "norm.weight":
        k = "final_norm.weight"

    return k

mapped = {rename_key(k): v for k, v in hf_sd.items()}

# the file has no lm_head (tied!), but your state_dict lists out_head.weight
mapped["out_head.weight"] = mapped["tok_emb.weight"]

missing, unexpected = model.load_state_dict(mapped, strict=True)

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tok = AutoTokenizer.from_pretrained(REPO)
hf_model = AutoModelForCausalLM.from_pretrained(REPO, torch_dtype=torch.float32).eval()
model.eval()

ids = tok("The capital of France is", return_tensors="pt").input_ids
with torch.no_grad():
    mine = model(ids)
    ref  = hf_model(ids).logits

print((mine - ref).abs().max())                     # expect ~1e-4 or smaller
print(tok.decode(mine[0, -1].argmax()))             # expect " Paris" (or similar)