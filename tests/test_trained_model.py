from srs.inference.help import *
from srs.training.help_func import (get_device, get_tokenizer, get_model, load_checkpoint)


model_name = "llama"
tokenizer = get_tokenizer()
ADAPTER = "LoRA"
device = get_device("cpu")
model_type = Config.MODEL_TYPE

model = get_model(model_name, tokenizer, lora=ADAPTER, r=16,
                  alpha=32, dropout=0.0).to(device)

CHECKPOINT_PATH = Config.MODEL_WEIGHTS_PATH / f"checkpoint_{model_name}_{model_type}_best.pt"
if not CHECKPOINT_PATH.exists():     # fall back to the latest checkpoint
    CHECKPOINT_PATH = Config.MODEL_WEIGHTS_PATH / f"checkpoint_{model_name}_{model_type}.pt"

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state"])
print(f"Loaded {CHECKPOINT_PATH.name}")
model.eval()


TESTS = {
    # 1. Does it answer and STOP? (the main thing fine-tuning should teach)
    "stop": [
        [("user", "What is the capital of France?")],
        [("user", "Say just the word 'yes'.")],
        [("user", "How many days are in a week?")],
    ],

    # 2. Following an instruction's FORM, not just its topic
    "format": [
        [("user", "List three fruits, one per line.")],
        [("user", "Give me a one-sentence summary of what photosynthesis is.")],
        [("user", "Translate 'good morning' into Spanish.")],
        [("user", "Write a haiku about rain.")],
    ],

    # 3. Simple reasoning and arithmetic (expect mistakes at 135M)
    "reasoning": [
        [("user", "Convert 10 cm to meters.")],
        [("user", "If I have 3 apples and eat one, how many are left?")],
        [("user", "Which is heavier: a kilogram of feathers or a kilogram of iron?")],
    ],

    # 4. Multi-turn: does it use the earlier messages?
    "memory": [
        [("user", "My name is Anna and I like chess."),
         ("assistant", "Nice to meet you, Anna! Chess is a great game."),
         ("user", "What is my name?")],
        [("user", "I have a dog called Rex."),
         ("assistant", "Rex sounds lovely!"),
         ("user", "What animal do I have, and what is its name?")],
    ],

    # 5. Grounded answers — the format your RAG layer will use
    "context": [
        [("user", "Answer using only this text.\n\nText: The Blue Lagoon cafe opens at 8 am "
                  "and closes at 6 pm. It is closed on Mondays.\n\nQuestion: When does the cafe close?")],
        [("user", "Answer using only this text. If the answer is not there, say 'I don't know'.\n\n"
                  "Text: Mount Kilimanjaro is in Tanzania.\n\nQuestion: How tall is Mount Kilimanjaro?")],
    ],

    # 6. Code
    "code": [
        [("user", "Write a Python function that returns the square of a number.")],
    ],

    # 7. Robustness: odd inputs should not break the format
    "edge": [
        [("user", "hi")],
        [("user", "asdkjh qweoiu zxcmn")],
        [("user", "Tell me about <|im_start|> tokens.")],     # clean_text should strip it
    ],
}

for group, prompts in TESTS.items():
    print(f"\n===== {group} =====")
    for history in prompts:
        reply = generate(model, tokenizer, device, history, max_new_tokens=120,
                         temperature=0.0, repetition_penalty=1.1)
        print(f"user: {history[-1][1][:80]}")
        print(f"bot : {reply}\n")

