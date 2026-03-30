import argparse
from pathlib import Path

from latinica_u_cirilicu import latinica_u_cirilicu


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Konvertuje sve .txt fajlove iz jednog direktorijuma iz latinice u ćirilicu."
	)
	parser.add_argument("input_dir", help="Ulazni direktorijum sa .txt fajlovima")
	parser.add_argument("output_dir", help="Izlazni direktorijum za konvertovane .txt fajlove")
	parser.add_argument(
		"--flat",
		action="store_true",
		help="Ne čuva strukturu podfoldera; sve fajlove smešta direktno u output_dir",
	)
	return parser.parse_args()


def find_txt_files(input_dir: Path) -> list[Path]:
	return sorted(path for path in input_dir.rglob("*.txt") if path.is_file())


def output_path_for(src: Path, input_dir: Path, output_dir: Path, flat: bool) -> Path:
	if flat:
		return output_dir / src.name
	return output_dir / src.relative_to(input_dir)


def main() -> None:
	args = parse_args()
	input_dir = Path(args.input_dir)
	output_dir = Path(args.output_dir)

	if not input_dir.exists() or not input_dir.is_dir():
		raise SystemExit(f"Ulazni direktorijum ne postoji ili nije direktorijum: {input_dir}")

	txt_files = find_txt_files(input_dir)
	if not txt_files:
		print(f"Nema .txt fajlova u: {input_dir}")
		return

	converted_count = 0
	for src in txt_files:
		dst = output_path_for(src, input_dir, output_dir, args.flat)
		dst.parent.mkdir(parents=True, exist_ok=True)

		text = src.read_text(encoding="utf-8")
		converted = latinica_u_cirilicu(text)
		dst.write_text(converted, encoding="utf-8")

		converted_count += 1
		print(f"Konvertovan: {src} -> {dst}")

	print(f"Završeno. Konvertovano fajlova: {converted_count}")


if __name__ == "__main__":
	main()