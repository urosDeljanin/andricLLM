import argparse
from pathlib import Path


def extract_after_delimiter(
	input_path: Path,
	output_path: Path,
	delimiter: str,
	keep_without_delimiter: bool,
	strip_output: bool,
) -> tuple[int, int]:
	output_path.parent.mkdir(parents=True, exist_ok=True)

	total_lines = 0
	written_lines = 0

	with input_path.open("r", encoding="utf-8", errors="replace") as in_file, output_path.open(
		"w", encoding="utf-8"
	) as out_file:
		for raw_line in in_file:
			total_lines += 1
			line = raw_line.rstrip("\n")

			if delimiter in line:
				part = line.split(delimiter, 1)[1]
				if strip_output:
					part = part.strip()
				out_file.write(part + "\n")
				written_lines += 1
			elif keep_without_delimiter:
				out_file.write("\n")
				written_lines += 1

	return total_lines, written_lines


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Napravi novi fajl sa delom svake linije koji dolazi posle delimitera."
	)
	parser.add_argument("input", type=Path, help="Putanja do ulaznog fajla")
	parser.add_argument("output", type=Path, help="Putanja do izlaznog fajla")
	parser.add_argument(
		"--delimiter",
		default=">>",
		help="Delimiter posle kog se uzima drugi deo linije (podrazumevano: >>)",
	)
	parser.add_argument(
		"--keep-without-delimiter",
		action="store_true",
		help="Ako linija nema delimiter, upisi praznu liniju da se zadrzi broj redova",
	)
	parser.add_argument(
		"--no-strip",
		action="store_true",
		help="Ne uklanjaj vodece i pratece razmake iz izdvojenog teksta",
	)

	args = parser.parse_args()

	if not args.input.exists():
		raise FileNotFoundError(f"Ulazni fajl ne postoji: {args.input}")

	total, written = extract_after_delimiter(
		input_path=args.input,
		output_path=args.output,
		delimiter=args.delimiter,
		keep_without_delimiter=args.keep_without_delimiter,
		strip_output=not args.no_strip,
	)

	print(f"Procitano linija: {total}")
	print(f"Upisano linija: {written}")
	print(f"Izlazni fajl: {args.output}")


if __name__ == "__main__":
	main()