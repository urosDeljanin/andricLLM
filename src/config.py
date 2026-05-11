from dataclasses import dataclass

import torch


@dataclass
class TrainConfig:
	model_path: str = "tokenizer/andric_sp.model"
	input_path: str = "input/sveKnjige.txt"
	save_path: str = "models/andric_model.pt"
	init_from: str | None = None
	block_size: int = 512
	batch_size: int = 20
	embed_dim: int = 384
	num_layers: int = 6
	num_heads: int = 6
	dropout: float = 0.1
	lr: float = 3e-4
	max_steps: int = 3000
	warmup_steps: int = 300
	eval_every: int = 50
	acc_steps: int = 25
	val_split: float = 0.1
	save_last_model: bool = False
	device: str = "cuda" if torch.cuda.is_available() else "cpu"
	seed: int = 42
