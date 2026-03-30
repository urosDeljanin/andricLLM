import argparse
from pathlib import Path


def merge_books(books_dir: Path, output_file: Path) -> None:
	txt_files = sorted(books_dir.glob("*.txt"))
	with output_file.open("w", encoding="utf-8") as out_f:
		for txt_file in txt_files:
			with txt_file.open("r", encoding="utf-8") as in_f:
				for raw_line in in_f:
					line = raw_line.replace("\x00", "").strip()
					if line:
						out_f.write(line + "\n")


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Merge all .txt files from books into input/sveKnjige.txt."
	)
	parser.add_argument(
		"--books",
		default="books",
		help="Path to the books directory.",
	)
	parser.add_argument(
		"--output",
		default="input/sveKnjige.txt",
		help="Output file path.",
	)
	args = parser.parse_args()

	books_dir = Path(args.books)
	output_file = Path(args.output)
	merge_books(books_dir, output_file)


if __name__ == "__main__":
	main()
