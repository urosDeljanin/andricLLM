import argparse

import torch

from .config import TrainConfig
from .generate import generate_text
from .train import train


def parse_args():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Train a transformer on Andric text using SentencePiece tokens.")
	
	general_group = parser.add_argument_group("General Arguments")
	general_group.add_argument("--mode", choices=["train", "generate"], default="train", help="Mode to run the script in.")
	general_group.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda/cpu).")
	general_group.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
	
	train_group = parser.add_argument_group("Training Arguments")
	train_group.add_argument("--model-path", default="tokenizer/andric_sp.model", help="Path to the SentencePiece model.")
	train_group.add_argument("--input-path", default="input/sveKnjige.txt", help="Path to the training text data.")
	train_group.add_argument("--save-path", default="models/andric_model.pt", help="Path to save the trained model.")
	train_group.add_argument("--init-from", default=None, help="Path to a pre-trained model to initialize from.")
	train_group.add_argument("--block-size", type=int, default=512, help="Context block size.")
	train_group.add_argument("--batch-size", type=int, default=20, help="Batch size per training step.")
	train_group.add_argument("--embed-dim", type=int, default=384, help="Embedding dimension size.")
	train_group.add_argument("--num-layers", type=int, default=6, help="Number of transformer layers.")
	train_group.add_argument("--num-heads", type=int, default=6, help="Number of attention heads.")
	train_group.add_argument("--dropout", type=float, default=0.1, help="Dropout probability.")
	train_group.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
	train_group.add_argument("--max-steps", type=int, default=3000, help="Maximum number of training steps.")
	train_group.add_argument("--warmup-steps", type=int, default=300, help="Number of warmup steps for learning rate scheduler.")
	train_group.add_argument("--eval-every", type=int, default=50, help="Interval for evaluation and saving checkpoints.")
	train_group.add_argument("--acc-steps", type=int, default=25, help="Gradient accumulation steps.")
	train_group.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio.")
	train_group.add_argument("--save-last-model", action="store_true", help="Save the model at the very end of training.")
	
	gen_group = parser.add_argument_group("Generation Arguments")
	gen_group.add_argument("--prompt", default="", help="Prompt to start text generation.")
	gen_group.add_argument("--max-new-tokens", type=int, default=500, help="Maximum number of new tokens to generate.")
	gen_group.add_argument("--temperature", type=float, default=0.7, help="Temperature for softmax sampling.")
	gen_group.add_argument("--top-k", type=int, default=30, help="Top-K sampling parameter.")
	
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
