import argparse

import torch

from .config import TrainConfig
from .generate import generate_text
from .train import train


def parse_args():
	parser = argparse.ArgumentParser(description="Train a transformer on Andric text using SentencePiece tokens.")
	parser.add_argument("--mode", choices=["train", "generate"], default="train")
	parser.add_argument("--model-path", default="tokenizer/andric_sp.model")
	parser.add_argument("--input-path", default="input/sveKnjige.txt")
	parser.add_argument("--save-path", default="models/andric_transformer.pt")
	parser.add_argument("--init-from", default=None)
	parser.add_argument("--block-size", type=int, default=256)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--embed-dim", type=int, default=384)
	parser.add_argument("--num-layers", type=int, default=6)
	parser.add_argument("--num-heads", type=int, default=6)
	parser.add_argument("--dropout", type=float, default=0.1)
	parser.add_argument("--lr", type=float, default=3e-4)
	parser.add_argument("--max-steps", type=int, default=3000)
	parser.add_argument("--warmup-steps", type=int, default=300)
	parser.add_argument("--eval-every", type=int, default=50)
	parser.add_argument("--acc-steps", type=int, default=25)
	parser.add_argument("--val-split", type=float, default=0.1)
	parser.add_argument("--save-last-model", action="store_true")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
	parser.add_argument("--prompt", default="")
	parser.add_argument("--max-new-tokens", type=int, default=1000)
	parser.add_argument("--temperature", type=float, default=0.5)
	parser.add_argument("--top-k", type=int, default=30)
	return parser.parse_args()


def main():
	args = parse_args()
	if args.mode == "train":
		cfg = TrainConfig(
			model_path=args.model_path,
			input_path=args.input_path,
			save_path=args.save_path,
			init_from=args.init_from,
			block_size=args.block_size,
			batch_size=args.batch_size,
			embed_dim=args.embed_dim,
			num_layers=args.num_layers,
			num_heads=args.num_heads,
			dropout=args.dropout,
			lr=args.lr,
			max_steps=args.max_steps,
			warmup_steps=args.warmup_steps,
			eval_every=args.eval_every,
			acc_steps=args.acc_steps,
			val_split=args.val_split,
			save_last_model=args.save_last_model,
			device=args.device,
			seed=args.seed,
		)
		train(cfg)
	else:
		generate_text(
			model_path=args.init_from,
			prompt=args.prompt,
			max_new_tokens=args.max_new_tokens,
			temperature=args.temperature,
			top_k=args.top_k,
			device=args.device,
		)


if __name__ == "__main__":
	main()
