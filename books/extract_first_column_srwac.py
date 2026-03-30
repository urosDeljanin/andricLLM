import argparse
import re
from pathlib import Path


def should_skip_line(line: str) -> bool:
	stripped = line.strip()
	if not stripped:
		return True
	if stripped.startswith("<") and stripped.endswith(">"):
		return True
	return False


PUNCT_NO_LEADING_SPACE = re.compile(r"^[,.;:!?%\)\]\}]+$")


def is_no_leading_space_token(token: str) -> bool:
	return bool(PUNCT_NO_LEADING_SPACE.match(token))


def append_token(current: str, token: str, glue_to_previous: bool) -> str:
	if not current:
		return token
	if glue_to_previous or is_no_leading_space_token(token):
		return current + token
	return current + " " + token


def extract_first_column_from_xml_folder(input_dir: Path, output_file: Path) -> tuple[int, int, int, int, int]:
	if not input_dir.exists():
		raise FileNotFoundError(f"Ulazni folder ne postoji: {input_dir}")
	if not input_dir.is_dir():
		raise NotADirectoryError(f"Ulazna putanja nije folder: {input_dir}")

	xml_files = sorted(input_dir.rglob("*.xml"))
	if not xml_files:
		raise FileNotFoundError(f"Nema XML fajlova u folderu: {input_dir}")

	output_file.parent.mkdir(parents=True, exist_ok=True)

	total_files = 0
	total_lines = 0
	written_tokens = 0
	written_sentences = 0
	written_paragraphs = 0

	with output_file.open("w", encoding="utf-8") as out:
		for xml_path in xml_files:
			total_files += 1
			in_sentence = False
			in_paragraph = False
			current_sentence = ""
			current_paragraph: list[str] = []
			glue_next = False

			with xml_path.open("r", encoding="utf-8", errors="replace") as inp:
				for raw_line in inp:
					total_lines += 1
					stripped = raw_line.strip()

					if stripped.startswith("<p") and stripped.endswith(">"):
						in_paragraph = True
						current_paragraph = []
						continue

					if stripped == "</p>":
						if in_sentence and current_sentence:
							current_paragraph.append(current_sentence)
							written_sentences += 1
							in_sentence = False
							current_sentence = ""
							glue_next = False

						if in_paragraph and current_paragraph:
							out.write(" ".join(current_paragraph) + "\n\n")
							written_paragraphs += 1

						in_paragraph = False
						current_paragraph = []
						continue

					if stripped == "<s>":
						in_sentence = True
						current_sentence = ""
						glue_next = False
						continue

					if stripped == "</s>":
						if in_sentence and current_sentence:
							current_paragraph.append(current_sentence)
							written_sentences += 1
						in_sentence = False
						current_sentence = ""
						glue_next = False
						continue

					if not in_sentence:
						continue

					if stripped == "<g/>":
						glue_next = True
						continue

					if should_skip_line(raw_line):
						continue

					parts = stripped.split()
					if not parts:
						continue

					token = parts[0]
					current_sentence = append_token(current_sentence, token, glue_next)
					glue_next = False
					written_tokens += 1

			# Ako fajl zavrsi bez zatvarajucih tagova, sacuvaj sto je skupljeno.
			if in_sentence and current_sentence:
				current_paragraph.append(current_sentence)
				written_sentences += 1

			if current_paragraph:
				out.write(" ".join(current_paragraph) + "\n\n")
				written_paragraphs += 1

	return total_files, total_lines, written_tokens, written_sentences, written_paragraphs


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Izvlaci reci iz prve kolone i sklapa ih u recenice iz svih XML fajlova u folderu."
	)
	parser.add_argument(
		"--input-dir",
		type=Path,
		default=Path("books/srWaC"),
		help="Folder sa XML fajlovima (podrazumevano: books/srWaC)",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("input/srWac_prva_kolona.txt"),
		help="Izlazni TXT fajl (podrazumevano: input/srWac_prva_kolona.txt)",
	)

	args = parser.parse_args()

	files, lines, tokens, sentences, paragraphs = extract_first_column_from_xml_folder(args.input_dir, args.output)

	print(f"Obradjeno XML fajlova: {files}")
	print(f"Procitano linija: {lines}")
	print(f"Upisano tokena (prva kolona): {tokens}")
	print(f"Upisano recenica: {sentences}")
	print(f"Upisano pasusa: {paragraphs}")
	print(f"Izlazni fajl: {args.output}")


if __name__ == "__main__":
	main()