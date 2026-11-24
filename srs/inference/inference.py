# inference.py
"""Chat with the fine-tuned SmolLM2 (Llama architecture) in the terminal.

Run from the project root:
    python -m srs.inference.inference

The adapter type (None / LoRA / DoRA) and its rank r are read from the checkpoint itself,
so they always match what was trained. Only LORA_ALPHA has to match training by hand.
"""
import torch

from srs.LLM.config import Config
from srs.training.help_func import get_device, get_tokenizer
from srs.inference.help import generate
from model.Llama_full_model import LlamaManager

# ---- generation settings
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7          # 0 = greedy (repeatable answers)
TOP_K = 40
TOP_P = 0.9
REPETITION_PENALTY = 1.1

# ---- must match LORA_ALPHA in srs/training/new_training.py (scaling = alpha / r)
LORA_ALPHA = 32

MODEL_NAME = "llama"
MODEL_TYPE = Config.MODEL_TYPE


def find_checkpoint():
    """Best checkpoint if it exists, otherwise the latest one."""
    best = Config.MODEL_WEIGHTS_PATH / f"checkpoint_{MODEL_NAME}_{MODEL_TYPE}_best.pt"
    latest = Config.MODEL_WEIGHTS_PATH / f"checkpoint_{MODEL_NAME}_{MODEL_TYPE}.pt"
    for path in (best, latest):
        if path.exists():
            return path
    raise FileNotFoundError(f"no checkpoint found: {best.name} / {latest.name} in {Config.MODEL_WEIGHTS_PATH}")


def detect_adapter(state_dict):
    """Read the adapter type and rank from the parameter names and shapes in the checkpoint."""
    if any(k.endswith(".magnitude") for k in state_dict):
        adapter = "DoRA"
    elif any(k.endswith(".A") for k in state_dict):
        adapter = "LoRA"
    else:
        return None, None                     # full fine-tune
    a_key = next(k for k in state_dict if k.endswith(".A"))
    r = state_dict[a_key].shape[0]            # A has shape (r, in_features)
    return adapter, r


def load_model(device, tokenizer):
    path = find_checkpoint()
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint["model_state"]

    adapter, r = detect_adapter(state)
    manager = LlamaManager(use_lora=adapter, r=r or 8, alpha=LORA_ALPHA, dropout=0.0)
    # inference=True: skip downloading the pretrained weights, the checkpoint contains all of them
    model = manager.get_model(tokenizer=tokenizer, inference=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    info = f"adapter={adapter}" + (f", r={r}, alpha={LORA_ALPHA}" if adapter else "")
    if "val_loss" in checkpoint:
        info += f", val loss {checkpoint['val_loss']:.4f}"
    if "global_step" in checkpoint:
        info += f", step {checkpoint['global_step']}"
    print(f"Loaded {path.name} ({info}) on {device}")
    return model


def main():
    device = get_device()
    tokenizer = get_tokenizer()
    model = load_model(device, tokenizer)

    print("Chat started. Type 'clear' to clear history. Type 'exit' to quit.\n")
    history = []

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            break
        if user_input.lower() == "clear":
            history.clear()
            print("Conversation history erased. Starting fresh!\n")
            continue

        history.append(("user", user_input))
        # the history is formatted with the same ChatML template as the training data
        answer = generate(model, tokenizer, device, history,
                          max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE,
                          top_k=TOP_K, top_p=TOP_P, repetition_penalty=REPETITION_PENALTY)
        print(f"Bot: {answer}\n")
        history.append(("assistant", answer))


if __name__ == "__main__":
    main()
