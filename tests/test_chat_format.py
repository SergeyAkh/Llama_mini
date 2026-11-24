"""Tests for the ChatML chat format and label masking (step 8c).

Run from the project root:   python -m tests.test_chat_format
"""
from transformers import AutoTokenizer

from dataset.chat_format import (IM_START, IM_END, format_conversation,
                                 build_prompt, compute_labels, truncate_to_last_complete_turn)



IGNORE = -100
tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
HEADER = tok.encode(IM_START + "assistant\n", add_special_tokens=False)
END = tok.convert_tokens_to_ids(IM_END)
NEWLINE = tok.encode("\n", add_special_tokens=False)[0]

failures = 0


def check(cond, name):
    global failures
    print(("PASS  " if cond else "FAIL  ") + name)
    if not cond:
        failures += 1


def encode(text):
    return tok.encode(text, add_special_tokens=False)


def trained_text(ids, labels):
    """Decode only the tokens that get a training signal."""
    return tok.decode([t for t, l in zip(ids, labels) if l != IGNORE])


# ---------------------------------------------------------------- basics
check(HEADER == [1, 520, 9531, 198], f"assistant header is 4 tokens: {HEADER}")
check(END == 2, f"<|im_end|> id is 2 (got {END})")

# ---------------------------------------------------------------- one turn
ids = encode(format_conversation([("Hi", "Hello!")]))
labels = compute_labels(ids, HEADER, END, IGNORE)

check(len(labels) == len(ids), "labels have the same length as ids")
check(all(l == IGNORE or l == t for t, l in zip(ids, labels)),
      "every label is either -100 or the token itself")
check(trained_text(ids, labels) == "Hello!" + IM_END,
      f"only the reply + <|im_end|> is trained: {trained_text(ids, labels)!r}")
check(labels[-1] == IGNORE and ids[-1] == NEWLINE,
      "the '\\n' after the final <|im_end|> is masked")

# ---------------------------------------------------------------- two turns
turns = [("Hi", "Hello!"), ("How are you?", "Fine,\nthanks.")]
ids = encode(format_conversation(turns))
labels = compute_labels(ids, HEADER, END, IGNORE)

check(trained_text(ids, labels) == "Hello!" + IM_END + "Fine,\nthanks." + IM_END,
      f"both replies trained, nothing else: {trained_text(ids, labels)!r}")
check(sum(1 for l in labels if l == END) == len(turns),
      "exactly one trained <|im_end|> per assistant reply")
check(sum(1 for t in ids if t == END) == 2 * len(turns),
      "sanity: the text has one <|im_end|> per message (user + assistant)")

# the '\n' inside a reply is trained, the '\n' right after <|im_end|> is not
newline_positions = [i for i, t in enumerate(ids) if t == NEWLINE]
inside_reply = [i for i in newline_positions if ids[i - 1] != END and labels[i] != IGNORE]
after_end = [i for i in newline_positions if ids[i - 1] == END]
check(len(inside_reply) == 1, "the newline inside 'Fine,\\nthanks.' is trained")
check(all(labels[i] == IGNORE for i in after_end), "every newline after <|im_end|> is masked")

# header tokens are never trained
for start in (i for i in range(len(ids)) if ids[i:i + len(HEADER)] == HEADER):
    check(all(labels[j] == IGNORE for j in range(start, start + len(HEADER))),
          f"header at position {start} is fully masked")

# no user text leaks into training
check("Hi" not in trained_text(ids, labels).replace("Hello", "")
      and "How are you" not in trained_text(ids, labels),
      "user messages are never trained")

# ---------------------------------------------------------------- prompt vs training text
history = [("user", "Hi"), ("assistant", "Hello!"), ("user", "How are you?")]
prompt_ids = encode(build_prompt(history))
check(ids[:len(prompt_ids)] == prompt_ids,
      "build_prompt() tokens are an exact prefix of the training tokens")
first_reply_end = labels.index(END)
check(all(l == IGNORE for l in labels[first_reply_end + 1:len(prompt_ids)]),
      "everything between the first reply and the end of the prompt is masked")

# ---------------------------------------------------------------- edge cases
ids = encode(build_prompt([("user", "Hi")]))
labels = compute_labels(ids, HEADER, END, IGNORE)
check(all(l == IGNORE for l in labels),
      "an open prompt (header without a reply) trains nothing")

ids = encode(format_conversation([("Who are you?", "I am an assistant.")]))
labels = compute_labels(ids, HEADER, END, IGNORE)
check(trained_text(ids, labels) == "I am an assistant." + IM_END,
      "the word 'assistant' inside a reply does not restart the header logic")

ids = encode(format_conversation([("Say assistant", "OK")]))
labels = compute_labels(ids, HEADER, END, IGNORE)
check(trained_text(ids, labels) == "OK" + IM_END,
      "the word 'assistant' inside a user message is not mistaken for a header")

ids = encode(format_conversation([("Hi", "")]))
labels = compute_labels(ids, HEADER, END, IGNORE)
check(trained_text(ids, labels) == IM_END,
      "an empty reply still trains the <|im_end|> (the model learns to stop)")

ids = []
check(compute_labels(ids, HEADER, END, IGNORE) == [], "empty input gives empty labels")

ids, labels = truncate_to_last_complete_turn(ids, labels, END, 512)

# ---------------------------------------------------------------- summary
print()
print("ALL PASSED" if failures == 0 else f"{failures} FAILED")
