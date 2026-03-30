import argparse
from pathlib import Path


SINGLE_CHAR_MAP = {
	"a": "а",
	"b": "б",
	"v": "в",
	"g": "г",
	"d": "д",
	"đ": "ђ",
	"e": "е",
	"ž": "ж",
	"z": "з",
	"i": "и",
	"j": "ј",
	"k": "к",
	"l": "л",
	"m": "м",
	"n": "н",
	"o": "о",
	"p": "п",
	"r": "р",
	"s": "с",
	"t": "т",
	"ć": "ћ",
	"u": "у",
	"f": "ф",
	"h": "х",
	"c": "ц",
	"č": "ч",
	"š": "ш",
}

DIGRAPH_MAP = {
	"nj": "њ",
	"lj": "љ",
	"dž": "џ",
	"dz": "џ",
}


def _map_single_char(ch: str) -> str:
	lower = ch.lower()
	mapped = SINGLE_CHAR_MAP.get(lower)
	if mapped is None:
		return ch
	if ch.isupper():
		return mapped.upper()
	return mapped


def _map_digraph(chunk: str) -> str | None:
	lower = chunk.lower()
	mapped = DIGRAPH_MAP.get(lower)
	if mapped is None:
		return None
	if chunk.isupper():
		return mapped.upper()
	if chunk[0].isupper() and chunk[1].islower():
		return mapped.upper()
	return mapped


def latinica_u_cirilicu(text: str) -> str:
	result: list[str] = []
	i = 0
	while i < len(text):
		if i + 1 < len(text):
			chunk = text[i : i + 2]
			digraph_mapped = _map_digraph(chunk)
			if digraph_mapped is not None:
				result.append(digraph_mapped)
				i += 2
				continue
		result.append(_map_single_char(text[i]))
		i += 1
	return "".join(result)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Pretvaranje txt fajla iz srpske latinice u srpsku cirilicu.")
	parser.add_argument("input", help="Putanja do ulaznog .txt fajla")
	parser.add_argument(
		"-o",
		"--output",
		default=None,
		help="Putanja izlaznog .txt fajla (default: <ime>-cirilica.txt)",
	)
	parser.add_argument(
		"--chunk-size",
		type=int,
		default=1024 * 1024,
		help="Velicina chunk-a za obradu velikih fajlova (default: 1048576)",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	input_path = Path(args.input)
	output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}-cirilica.txt")
	chunk_size = max(2, args.chunk_size)

	carry = ""
	with input_path.open("r", encoding="utf-8") as infile, output_path.open("w", encoding="utf-8") as outfile:
		while True:
			chunk = infile.read(chunk_size)
			if not chunk:
				break

			data = carry + chunk
			if len(data) == 1:
				carry = data
				continue

			to_process = data[:-1]
			carry = data[-1]
			outfile.write(latinica_u_cirilicu(to_process))

		if carry:
			outfile.write(latinica_u_cirilicu(carry))

	print(f"Sacuvan fajl: {output_path}")


if __name__ == "__main__":
	main()
