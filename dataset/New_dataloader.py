"""Dataloader for chat fine-tuning.

Differences from the first version:
  * conversations are never cut in the middle of an answer: an over-long conversation is
    shortened to its last complete assistant reply that fits, and dropped if even the first
    reply does not fit (so the closing <|im_end|> is always present)
  * labels: only assistant replies (plus the <|im_end|> closing each one) are trained;
    user text, the <|im_start|>assistant\n header and the newline after <|im_end|> are masked
  * length-bucketed, shuffled batches (less padding, faster on a Mac)
"""
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from srs.LLM.config import Config
from dataset.chat_format import compute_labels, truncate_to_last_complete_turn
from dataset.batching import BucketBatchSampler

IGNORE_INDEX = Config.IGNORE_INDEX


class ConversationDataset(Dataset):
    def __init__(self, conversations, tokenizer, header_ids, end_id,
                 max_length=512, ignore_index=IGNORE_INDEX, verbose=True):
        self.samples = []          # list of (input_ids, labels) python lists
        stats = {"kept": 0, "shortened_to_last_turn": 0, "dropped_too_long": 0,
                 "dropped_no_supervision": 0}

        chunk = 2000
        for start in tqdm(range(0, len(conversations), chunk), desc="tokenizing", disable=not verbose):
            batch_text = conversations[start:start + chunk]
            encoded = tokenizer(batch_text, add_special_tokens=False)["input_ids"]

            for ids in encoded:
                labels = compute_labels(ids, header_ids, end_id, ignore_index)

                if max_length and len(ids) > max_length:
                    result = truncate_to_last_complete_turn(ids, labels, end_id, max_length)
                    if result is None:
                        stats["dropped_too_long"] += 1
                        continue
                    ids, labels = result[0], result[1]
                    stats["shortened_to_last_turn"] += 1

                if all(l == ignore_index for l in labels):
                    stats["dropped_no_supervision"] += 1
                    continue
                self.samples.append((ids, labels))
                stats["kept"] += 1

        self.lengths = [len(ids) for ids, _ in self.samples]
        self.stats = stats
        if verbose:
            n = len(conversations)
            print(f"[dataloader] {n} conversations -> kept {stats['kept']} "
                  f"(shortened {stats['shortened_to_last_turn']}, dropped too long {stats['dropped_too_long']}, "
                  f"dropped w/o supervision {stats['dropped_no_supervision']}); max_length={max_length}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids, labels = self.samples[idx]
        return {"input_ids": torch.tensor(ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)}


def collate_fn(batch, pad_token_id, ignore_index=IGNORE_INDEX):
    max_len = max(item["input_ids"].shape[0] for item in batch)
    bs = len(batch)
    input_ids = torch.full((bs, max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((bs, max_len), ignore_index, dtype=torch.long)
    attention_mask = torch.zeros((bs, max_len), dtype=torch.long)
    for i, item in enumerate(batch):
        n = item["input_ids"].shape[0]
        input_ids[i, :n] = item["input_ids"]
        labels[i, :n] = item["labels"]
        attention_mask[i, :n] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def create_dataloader(conversations, tokenizer, batch_size=8, shuffle=True, max_length=512,
                      num_workers=0, seed=0):
    """Same signature as before (+ seed). Returns a DataLoader; dataloader.batch_sampler.set_epoch(e)
    changes the shuffling per epoch."""
    header_ids = tokenizer.encode(Config.IM_START + "assistant\n", add_special_tokens=False)
    end_id = tokenizer.convert_tokens_to_ids(Config.IM_END)

    dataset = ConversationDataset(
        conversations=list(conversations), tokenizer=tokenizer,
        header_ids=header_ids, end_id=end_id, max_length=max_length,
    )
    sampler = BucketBatchSampler(dataset.lengths, batch_size, shuffle=shuffle, seed=seed)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=lambda b: collate_fn(b, pad_token_id=tokenizer.pad_token_id),
        pin_memory=torch.cuda.is_available(),
    )
