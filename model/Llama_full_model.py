from os import abort

from transformers import GPT2LMHeadModel

from srs.LLM.config import Config
# from model.Load_model import GPT2Loader
from model.Llama_manual_architecture import LlamaModel, GroupedQueryAttention, FeedForward
from model.LoRA import LoRALinear
from model.DoRA import DoRALinear
from model.Load_llama import LlamaWeightLoader
import torch
import torch.nn as nn

ATTN_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_TARGETS  = ("gate_proj", "up_proj", "down_proj")

class LlamaManager:

    def __init__(self, use_lora=None, r=8, alpha = 1, dropout=0.05):
        # Load configuration
        self.cfg = Config.LLAMA_CONFIG
        self.r, self.alpha, self.dropout = r, alpha, dropout
        self.use_lora = use_lora
        self.model = None


    def prepare_model(self, tokenizer, inference=None):
        # 1. sanity check: tokenizer and model must agree on the vocabulary
        assert len(tokenizer) == self.cfg["vocab_size"], \
            f"tokenizer has {len(tokenizer)} tokens, model expects {self.cfg['vocab_size']}"
        # 2. build the architecture
        self.model = LlamaModel(self.cfg)
        # 3. pretrained weights — only for training.
        #    For inference your checkpoint is loaded afterwards and overwrites them anyway.
        if inference is None:
            self.model = LlamaWeightLoader(Config.LLAMA_REPO).load(self.model)

        # 4. adapters + freezing (parts 2 and 3)
        self.train_token_ids = tokenizer.convert_tokens_to_ids([Config.IM_START, Config.IM_END])
        if self.use_lora in ("LoRA", "DoRA"):
            self.apply_adapters(self.model)
            self.freeze_except_adapter(self.model, self.train_token_ids)

        return self.model


    def get_model(self, tokenizer=None, inference=None):
        return self.prepare_model(tokenizer, inference)

    def freeze_except_adapter(self, model, train_token_ids):
        # 1. adapters only
        for name, param in model.named_parameters():
            param.requires_grad = name.endswith((".A", ".B", ".magnitude"))

        # 2. plus selected rows of the (shared) embedding matrix
        keep = torch.zeros(self.cfg["vocab_size"], dtype=torch.bool)
        keep[train_token_ids] = True

        def only_selected_rows(grad):
            return grad * keep.to(grad.device).unsqueeze(1).to(grad.dtype)

        W = model.tok_emb.weight
        W.requires_grad = True
        W.register_hook(only_selected_rows)

    def apply_adapters(self, model):
        adapter_cls = LoRALinear if self.use_lora == "LoRA" else DoRALinear

        for module in list(model.modules()):
            if isinstance(module, GroupedQueryAttention):
                names = ATTN_TARGETS
            elif isinstance(module, FeedForward):
                names = MLP_TARGETS
            else:
                continue

            for name in names:
                layer = getattr(module, name)
                if not isinstance(layer, nn.Linear):  # already wrapped → skip
                    continue
                wrapped = adapter_cls(layer, r=self.r, alpha=self.alpha, dropout=self.dropout)
                setattr(module, name, wrapped)