from dataclasses import dataclass

import torch


@dataclass
class TrainConfig:
	model_path: str = "tokenizer/sr_tokenizer.model"
	input_path: str = "input/datasetbpe.txt"
	save_path: str = "models/sr_model.pt"
	num_saved_models: int = 1
	init_from: str | None = None
	block_size: int = 1024
	batch_size: int = 16
	embed_dim: int = 768
	num_layers: int = 8
	num_heads: int = 12
	dropout: float = 0.1
	lr: float = 3e-4
	max_steps: int = 3000
	warmup_steps: int = 300
	eval_every: int = 50
	acc_steps: int = 1
	val_split: float = 0.05
	save_last_model: bool = True
	device: str = "cuda" if torch.cuda.is_available() else "cpu"
	seed: int = 42