import sentencepiece as spm
import torch

from .config import TrainConfig
from .model import TransformerLM
from .tokenization import encode_with_inline_eos


def _resolve_device(requested_device: str) -> str:
	if requested_device == "cuda" and not torch.cuda.is_available():
		return "cpu"
	return requested_device


def generate_text(
	model_path: str,
	prompt: str,
	max_new_tokens: int,
	temperature: float,
	top_k: int | None,
	device: str | None = None,
):
	try:
		ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
	except TypeError:
		# Fallback for older PyTorch versions that do not support weights_only.
		ckpt = torch.load(model_path, map_location="cpu")
	cfg = TrainConfig(**ckpt["config"])
	gen_device = _resolve_device(device or cfg.device)
	sp = spm.SentencePieceProcessor()
	sp.load(cfg.model_path)
	model = TransformerLM(ckpt["vocab_size"], cfg.block_size, cfg.embed_dim, cfg.num_layers, cfg.num_heads, cfg.dropout)
	model.load_state_dict(ckpt["model_state"])
	model.to(gen_device)
	model.eval()

	if prompt:
		prompt_ids = encode_with_inline_eos(prompt, sp)
	else:
		prompt_ids = [sp.bos_id()] if sp.bos_id() != -1 else [0]
	eos_id = sp.eos_id()
	if eos_id < 0:
		eos_id = None
	idx = torch.tensor(prompt_ids, dtype=torch.long, device=gen_device).unsqueeze(0)
	out_ids = model.generate(
		idx,
		max_new_tokens=max_new_tokens,
		temperature=temperature,
		top_k=top_k,
		eos_id=eos_id,
	)[0].tolist()
	text = sp.decode(out_ids)
	print(text)
