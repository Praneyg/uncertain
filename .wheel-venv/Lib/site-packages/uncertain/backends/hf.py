"""Local/reference Hugging Face execution; all optional imports occur at use."""
from hashlib import sha256
from importlib.metadata import version, PackageNotFoundError
import numpy as np
from uncertain.core import ModelSpec, ConfigError


class HFBackend:
    def __init__(self, model_id=None, *, revision=None, model=None, tokenizer=None,
                 bos_token_id=None, eos_token_id=None, device="cpu"):
        self._model, self._tokenizer = model, tokenizer
        self.device = device
        if model is not None:
            h = sha256()
            for name, tensor in sorted(model.state_dict().items()):
                h.update(name.encode())
                h.update(tensor.detach().cpu().float().numpy().tobytes())
            self.model_spec = ModelSpec("local-hf", h.hexdigest())
            self.bos_token_id = bos_token_id if bos_token_id is not None else model.config.bos_token_id
            self.eos_token_id = eos_token_id if eos_token_id is not None else model.config.eos_token_id
        else:
            self.model_spec = ModelSpec(model_id, revision)
            self.bos_token_id, self.eos_token_id = bos_token_id, eos_token_id
        tokenizer_identity = None
        if tokenizer is not None:
            from uncertain.utils import digest
            tokenizer_identity = digest({"vocab": tokenizer.get_vocab(),
                                         "special": tokenizer.special_tokens_map,
                                         "class": type(tokenizer).__name__})
        self.descriptor = {"backend": "hf-reference:v1", "model": self.model_spec,
                           "tokenizer": tokenizer_identity, "device": device,
                           "precision": "float32", "bos": self.bos_token_id,
                           "eos": self.eos_token_id, "batching": "one-sequence"}
        for package in ("torch", "transformers"):
            try:
                self.descriptor[package] = version(package)
            except PackageNotFoundError:
                self.descriptor[package] = None

    def _load(self):
        import torch
        if self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_spec.model_id, revision=self.model_spec.revision,
                attn_implementation="eager")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_spec.model_id, revision=self.model_spec.revision)
        self._model.to(device=self.device, dtype=torch.float32).eval()
        if self.bos_token_id is None:
            self.bos_token_id = self._model.config.bos_token_id
        if self.eos_token_id is None:
            eos = self._model.config.eos_token_id
            if isinstance(eos, list):
                raise ConfigError("reference backend requires a single explicit eos_token_id")
            self.eos_token_id = eos
        return torch

    def _prefix(self, context):
        if context:
            return tuple(context)
        if self.bos_token_id is None:
            raise ConfigError("empty contexts require an explicit BOS token")
        return (self.bos_token_id,)

    def probabilities(self, context):
        torch = self._load()
        with torch.inference_mode():
            ids = torch.tensor([self._prefix(context)], device=self.device)
            logits = self._model(ids).logits[0, -1].double()
            return torch.softmax(logits, dim=-1).cpu().numpy()

    def score(self, context, tokens):
        """Independent single teacher-forced forward, not the generation loop."""
        torch = self._load()
        if not tokens:
            return np.array([], dtype=float)
        prefix = self._prefix(context)
        with torch.inference_mode():
            ids = torch.tensor([prefix + tuple(tokens)], device=self.device)
            logits = self._model(ids).logits[0, len(prefix)-1:len(prefix)+len(tokens)-1].double()
            logp = torch.log_softmax(logits, dim=-1)
            target = torch.tensor(tokens, device=self.device)
            return logp[torch.arange(len(tokens), device=self.device), target].cpu().numpy()

    def encode(self, text):
        self._load_tokenizer()
        if self._tokenizer is None:
            raise ConfigError("text encoding requires a tokenizer; supply token IDs instead")
        return tuple(self._tokenizer.encode(text, add_special_tokens=False))

    def decode(self, tokens):
        self._load_tokenizer()
        if self._tokenizer is None:
            return " ".join(str(t) for t in tokens)
        return self._tokenizer.decode(tokens, skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)

    def _load_tokenizer(self):
        if self._tokenizer is None and self.model_spec.model_id != "local-hf":
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_spec.model_id, revision=self.model_spec.revision)

    def hidden(self, context, tokens, layer):
        torch = self._load()
        prefix = self._prefix(context)
        if not tokens:
            return np.empty((0, self._model.config.hidden_size))
        with torch.inference_mode():
            ids = torch.tensor([prefix + tuple(tokens)], device=self.device)
            hidden = self._model(ids, output_hidden_states=True).hidden_states
            if not -len(hidden) <= layer < len(hidden):
                raise ConfigError(f"layer {layer} outside {len(hidden)} available layers")
            return hidden[layer][0, len(prefix):].cpu().numpy().astype(float)
