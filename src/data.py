import sentencepiece as spm
import torch

from .config import TrainConfig
import os
import numpy as np


def _token_dtype(vocab_size: int):
	if vocab_size <= np.iinfo(np.uint16).max:
		return np.uint16
	return np.int32


def _cache_path(input_path: str, vocab_size: int, dtype: np.dtype) -> str:
	return input_path + f".{vocab_size}.{np.dtype(dtype).name}.bin"


def load_tokens(model_path: str, input_path: str):
	sp = spm.SentencePieceProcessor()
	sp.load(model_path)
	return sp, load_ids(sp, input_path)


def load_ids(sp: spm.SentencePieceProcessor, input_path: str) -> np.memmap:
	vocab_size = sp.get_piece_size()
	dtype = _token_dtype(vocab_size)
	cache_path = _cache_path(input_path, vocab_size, dtype)
	progress_every = 50_000
	if os.path.exists(cache_path):
		return np.memmap(cache_path, dtype=dtype, mode="r")

	total_tokens = 0
	with open(input_path, "r", encoding="utf-8") as f:
		with open(cache_path, "wb") as out:
			for line_no, line in enumerate(f, start=1):
				line_ids = sp.encode(line, out_type=int, add_bos=True, add_eos=True)
				chunk = np.asarray(line_ids, dtype=dtype)
				chunk.tofile(out)
				total_tokens += chunk.size
				if line_no % progress_every == 0:
					print(f"tokenized {line_no:,} lines | {total_tokens:,} tokens")

	if total_tokens == 0:
		raise SystemExit("Input text is empty after tokenization.")

	print(f"tokenization complete | {total_tokens:,} tokens cached at {cache_path}")

	return np.memmap(cache_path, dtype=dtype, mode="r", shape=(total_tokens,))


def load_train_val(sp: spm.SentencePieceProcessor, cfg: TrainConfig):
	train_ids = load_ids(sp, cfg.input_path)
	if not (0.0 < cfg.val_split < 1.0):
		raise SystemExit("--val-split must be between 0 and 1.")
	total = len(train_ids)
	cut = int(total * (1.0 - cfg.val_split))
	val_ids = train_ids[cut:]
	train_ids = train_ids[:cut]
	return train_ids, val_ids


def get_batch(data, block_size: int, batch_size: int, device: str):
	max_start = len(data) - block_size - 1
	starts = torch.randint(0, max_start + 1, (batch_size,), dtype=torch.int64).numpy()
	offsets = np.arange(block_size, dtype=np.int64)
	x_np = data[starts[:, None] + offsets[None, :]]
	y_np = data[starts[:, None] + offsets[None, :] + 1]
	x = torch.from_numpy(x_np.astype(np.int64, copy=False))
	y = torch.from_numpy(y_np.astype(np.int64, copy=False))
	return x.to(device), y.to(device)
