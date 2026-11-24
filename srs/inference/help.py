import torch
import torch.nn.functional as F
from srs.LLM.config import Config
from model.Llama_full_model import LlamaManager
from dataset.chat_format import build_prompt as _build_prompt, IM_START, IM_END, EOS_TOKEN


def get_device(fixed = None):
    if fixed is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    else:
        return torch.device(fixed)

def get_model(tokenizer, lora=True, r=8, alpha=8, dropout=0.5):
    manager = LlamaManager(use_lora=lora, r=r, alpha=alpha, dropout=dropout)
    model = manager.get_model(tokenizer=tokenizer, inference=True)
    return model


def build_prompt(history):
    """history: list of (role, text) with role 'user' / 'assistant', ending with a user message.
    Uses the SAME template as the training data (dataset/chat_format.py)."""
    return _build_prompt(history)


def _as_prompt(prompt_or_history):
    if isinstance(prompt_or_history, (list, tuple)):
        return build_prompt(prompt_or_history)
    text = str(prompt_or_history)
    if IM_START in text:          # already a formatted prompt
        return text
    return build_prompt([("user", text)])  # a plain user message


@torch.no_grad()
def generate(model, tokenizer, device, prompt_or_history, max_new_tokens=200, temperature=0.7,
             top_k=40, top_p=0.9, repetition_penalty=1.1, context_length=Config.LLAMA_CONFIG["context_length"], model_name="custom_gpt2"):
    """Sampling with temperature / top-k / top-p / repetition penalty. Stops at <|im_end|>.

    temperature <= 0 means greedy decoding. Returns the assistant reply as a string.
    """
    was_training = model.training
    model.eval()
    prompt = _as_prompt(prompt_or_history)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_tensor = torch.tensor([ids], dtype=torch.long, device=device)  # (1, T)

    eos_id = tokenizer.convert_tokens_to_ids(IM_END)
    blocked = [tokenizer.convert_tokens_to_ids(IM_START)]
    generated = []

    logits, caches = model(prompt_tensor, use_cache=True)

    for _ in range(max_new_tokens):
        if len(ids) >= context_length:
            break
        logits = logits[0, -1].float()

        logits[blocked] = float("-inf")           # role markers are never part of a reply
        if repetition_penalty and repetition_penalty != 1.0 and generated:
            prev = torch.tensor(sorted(set(generated)), device=logits.device)
            vals = logits[prev]
            logits[prev] = torch.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)

        if temperature is None or temperature <= 0:
            next_id = int(torch.argmax(logits))
        else:
            logits = logits / temperature
            if top_k and top_k > 0:
                kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
                logits[logits < kth] = float("-inf")
            if top_p and top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum > top_p
                remove[1:] = remove[:-1].clone()   # always keep the best token
                remove[0] = False
                logits[sorted_idx[remove]] = float("-inf")
            next_id = int(torch.multinomial(F.softmax(logits, dim=-1), 1))

        if next_id == eos_id:
            break
        next_input = torch.tensor([[next_id]], dtype=torch.long, device=device)
        logits, caches = model(next_input, kv_caches=caches,
                               start_pos=len(ids), use_cache=True)
        ids.append(next_id)
        generated.append(next_id)

    if was_training:
        model.train()
    return tokenizer.decode(generated).strip()


def temp_predict(model_name, model, prompt, tokenizer, device, temperature=0.7, max_new_tokens=200,
                 top_k=40, top_p=0.9, repetition_penalty=1.1):
    """Backward-compatible wrapper. `prompt` may be a plain user message, an already formatted prompt,
    or a chat history list of (role, text)."""
    return generate(model, tokenizer, device, prompt, max_new_tokens=max_new_tokens,
                    temperature=temperature, top_k=top_k, top_p=top_p,
                    repetition_penalty=repetition_penalty, model_name=model_name)


def greedy_predict(model, prompt, tokenizer, device, max_new_tokens=200):
    return generate(model, tokenizer, device, prompt, max_new_tokens=max_new_tokens,
                    temperature=0.0, repetition_penalty=1.1)



