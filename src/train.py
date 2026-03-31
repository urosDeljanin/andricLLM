import math
import os
import random
from datetime import datetime
from dataclasses import asdict

import sentencepiece as spm
import torch
from torch import nn

from .config import TrainConfig
from .data import get_batch, load_train_val
from .model import TransformerLM


@torch.no_grad()
def estimate_loss(model: nn.Module, data: torch.Tensor, cfg: TrainConfig):
	model.eval()
	losses = []
	for _ in range(10):
		xb, yb = get_batch(data, cfg.block_size, cfg.batch_size, cfg.device)
		_, loss = model(xb, yb)
		losses.append(loss.item())
	model.train()
	return sum(losses) / len(losses)


def train(cfg: TrainConfig):
	random.seed(cfg.seed)
	torch.manual_seed(cfg.seed)
	os.makedirs("output", exist_ok=True)
	log_path = os.path.join("output", f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

	def log(message: str) -> None:
		print(message)
		with open(log_path, "a", encoding="utf-8") as log_file:
			log_file.write(message + "\n")

	log(f"training log: {log_path}")
	log(f"config: {asdict(cfg)}")

	sp = spm.SentencePieceProcessor()
	sp.load(cfg.model_path)
	train_data, val_data = load_train_val(sp, cfg)
	vocab_size = sp.get_piece_size()
	model = TransformerLM(vocab_size, cfg.block_size, cfg.embed_dim, cfg.num_layers, cfg.num_heads, cfg.dropout)
	if cfg.init_from:
		if not os.path.exists(cfg.init_from):
			raise FileNotFoundError(f"Pretrained checkpoint not found: {cfg.init_from}")
		try:
			init_ckpt = torch.load(cfg.init_from, map_location="cpu", weights_only=True)
		except TypeError:
			# Fallback for older PyTorch versions that do not support weights_only.
			init_ckpt = torch.load(cfg.init_from, map_location="cpu")
		if "model_state" not in init_ckpt:
			raise KeyError(f"Checkpoint missing 'model_state': {cfg.init_from}")
		model.load_state_dict(init_ckpt["model_state"])
		log(f"initialized model weights from {cfg.init_from}")
	model.to(cfg.device)
	optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
	optimizer.zero_grad(set_to_none=True)
	best_val_loss = float("inf")
	best_state = None

	for step in range(1, cfg.max_steps + 1):
		xb, yb = get_batch(train_data, cfg.block_size, cfg.batch_size, cfg.device)
		_, loss = model(xb, yb)
		train_loss = loss.item()
		loss /= cfg.acc_steps
		loss.backward()

		if step % cfg.acc_steps == 0 or step == cfg.max_steps:
			optimizer.step()
			optimizer.zero_grad(set_to_none=True)

		if step % cfg.eval_every == 0 or step == 1:
			val_loss = estimate_loss(model, val_data, cfg)
			ppl = math.exp(val_loss)
			log(f"step {step} | train {train_loss:.4f} | val {val_loss:.4f} | ppl {ppl:.2f}")
			if val_loss < best_val_loss:
				best_val_loss = val_loss
				best_state = {k: v.cpu() for k, v in model.state_dict().items()}

	os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)
	if best_state is not None:
		model.load_state_dict(best_state)
	ckpt = {
		"model_state": model.state_dict(),
		"config": asdict(cfg),
		"vocab_size": vocab_size,
		"best_val_loss": best_val_loss,
	}
	torch.save(ckpt, cfg.save_path)
	log(f"best val loss: {best_val_loss:.4f}")
	log(f"saved to {cfg.save_path}")
