import argparse
from pathlib import Path

from pdfminer.high_level import extract_text


def convert_pdf_to_txt(pdf_path: Path, txt_path: Path) -> None:
	text = extract_text(str(pdf_path))
	txt_path.write_text(text, encoding="utf-8")


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Convert all PDF files from a folder into .txt files."
	)
	parser.add_argument(
		"--pdf-dir",
		default=str(Path("books") / "raw" / "pdf"),
		help="Folder with input PDF files.",
	)
	parser.add_argument(
		"--txt-dir",
		default=str(Path("books") / "raw" / "txt"),
		help="Folder where output .txt files are written.",
	)
	args = parser.parse_args()

	pdf_dir = Path(args.pdf_dir)
	txt_dir = Path(args.txt_dir)
	txt_dir.mkdir(parents=True, exist_ok=True)

	pdf_files = sorted(pdf_dir.glob("*.pdf"))
	if not pdf_files:
		raise SystemExit(f"No PDF files found in {pdf_dir}")

	for pdf_path in pdf_files:
		txt_path = txt_dir / f"{pdf_path.stem}.txt"
		convert_pdf_to_txt(pdf_path, txt_path)


if __name__ == "__main__":
	main()
