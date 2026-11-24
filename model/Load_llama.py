from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from model.Llama_manual_architecture import LlamaModel
from srs.LLM.config import Config

class LlamaWeightLoader:
    def __init__(self, repo_id=Config.LLAMA_REPO, filename="model.safetensors"):
        self.repo_id = repo_id
        self.filename = filename

    @staticmethod
    def rename_key(k):
        if k.startswith("model."):
            k = k[len("model."):]
        k = k.replace("embed_tokens", "tok_emb")
        if k.startswith("layers."):
            k = "trf_blocks." + k[len("layers."):]
        if k == "norm.weight":
            k = "final_norm.weight"
        return k

    def _load_hf_state_dict(self):
        w_path = hf_hub_download(self.repo_id, self.filename)
        return load_file(w_path)

    def _map_state_dict(self, hf_sd):
        mapped = {self.rename_key(k): v for k, v in hf_sd.items()}
        # tied embeddings: the checkpoint has no lm_head
        if "out_head.weight" not in mapped:
            mapped["out_head.weight"] = mapped["tok_emb.weight"]
        return mapped

    def load(self, model: LlamaModel, strict=True, tie_weights=True):
        hf_sd = self._load_hf_state_dict()
        mapped = self._map_state_dict(hf_sd)
        model.load_state_dict(mapped, strict=strict)

        if tie_weights:
            # make out_head share the same tensor as tok_emb (true tying)
            model.out_head.weight = model.tok_emb.weight

        return model