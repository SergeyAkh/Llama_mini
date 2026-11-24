"""Single source of truth for the chat template and label masking.

Used by data preparation, the dataloader, training previews and inference, so the
format the model is trained on is exactly the format it is prompted with.
This module has no torch / transformers imports on purpose (easy to unit-test).

Template (one turn):
    <|user|>\n{user}\n<|assistant|>\n{assistant}<|endoftext|>
A conversation is the concatenation of its turns. <|endoftext|> closes EVERY
assistant reply (not only the last one), so the model learns where a turn ends.
"""

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
EOS_TOKEN = "<|endoftext|>"
SPECIAL_STRINGS = (IM_START, IM_END, EOS_TOKEN)


def clean_text(text) -> str:
    """Normalise a message and make sure it cannot inject special tokens."""
    if text is None or (isinstance(text, float) and text != text):  # None / NaN
        return ""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")

    return text.strip()


def format_message(role: str, text: str) -> str:
    return f"{IM_START}{role}\n{text}{IM_END}\n"


def format_turn(user: str, assistant: str) -> str:
    return format_message("user", user) + format_message("assistant", assistant)

def format_conversation(turns) -> str:
    """turns: list of (user_text, assistant_text) pairs."""
    return "".join(format_turn(u, a) for u, a in turns)

def build_prompt(history) -> str:
    """Prompt for generation.

    history: list of (role, text), role in {"user", "assistant"}, last item must be
    a user message. The result ends with '<|assistant|>\\n', ready for the model
    to continue.
    """
    if not history or history[-1][0] != "user":
        raise ValueError("history must be non-empty and end with a user message")
    parts = []
    for role, text in history:
        text = clean_text(text)
        if role not in ("user", "assistant"):
            raise ValueError(f"unknown role: {role}")
        parts.append(format_message(role, text))

    parts.append(f"{IM_START}assistant\n")
    return "".join(parts)


def compute_labels(ids, header_ids, end_id, ignore_index=-100):
    """header_ids: token ids of "<|im_start|>assistant\\n" e.g. [1, 520, 9531, 198]
       end_id:     id of "<|im_end|>" e.g. 2"""
    labels = [ignore_index] * len(ids)
    n = len(header_ids)
    i = 0
    in_reply = False

    while i < len(ids):
        if not in_reply and list(ids[i : i + n]) == list(header_ids):
            in_reply = True
            i += n
            continue
        if in_reply:
            labels[i] = ids[i]
            if ids[i] == end_id:
                in_reply = False

        i += 1
    return labels


def truncate_to_last_complete_turn(ids, labels, end_id, max_len):

    if len(ids) <= max_len:
        return (list(ids), list(labels))
    cut = -1
    for i in range(max_len):
        if labels[i] == end_id:
            cut = i
    if cut < 0:
        return None
    return (ids[:cut+1], labels[:cut+1])

