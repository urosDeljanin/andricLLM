import argparse
from pathlib import Path

try:
	import regex as re  # type: ignore
	_HAS_REGEX = True
except ImportError:
	import re
	_HAS_REGEX = False


def _is_page_number_line(line: str) -> bool:
	stripped = line.strip()
	if not stripped:
		return False
	if stripped.isdigit() and len(stripped) <= 4:
		return True
	if _HAS_REGEX:
		if re.fullmatch(r"-\d{1,4}-", stripped):
			return True
		if re.fullmatch(r"\d{1,4}/\d{1,4}", stripped):
			return True
	else:
		if stripped.startswith("-") and stripped.endswith("-"):
			inner = stripped[1:-1]
			if inner.isdigit() and len(inner) <= 4:
				return True
		if "/" in stripped:
			left, right = stripped.split("/", 1)
			if left.isdigit() and right.isdigit() and len(left) <= 4 and len(right) <= 4:
				return True
	return False


def _remove_control_chars(text: str) -> str:
	return re.sub(r"[\x00-\x08\x0b\x0e-\x1f]", "", text)


def _normalize_inline_whitespace(text: str) -> str:
	text = text.replace("\u00a0", " ")
	text = text.replace("\u2009", " ")
	text = text.replace("\u202f", " ")
	text = text.replace("\u205f", " ")
	text = text.replace("\u3000", " ")
	if _HAS_REGEX:
		return re.sub(r"[^\S\n]+", " ", text)
	return re.sub(r"[ \t]+", " ", text)


def _is_chapter_line(line: str) -> bool:
	stripped = line.strip()
	if not stripped:
		return False
	if len(stripped) > 40:
		return False
	if _HAS_REGEX:
		if re.fullmatch(r"[IVXLCDM]+", stripped, flags=re.IGNORECASE):
			return True
		if re.fullmatch(r"(GLAVA|ГЛАВА)\s+[IVXLCDM]+", stripped, flags=re.IGNORECASE):
			return True
		if re.fullmatch(r"(GLAVA|ГЛАВА)\s+\d{1,3}", stripped, flags=re.IGNORECASE):
			return True
		if re.fullmatch(r"\p{Lu}+(?:[ \t]+\p{Lu}+)*", stripped):
			return True
	else:
		upper = stripped.upper()
		roman = set("IVXLCDM")
		if upper and all(ch in roman for ch in upper):
			return True
		if upper.startswith("GLAVA ") or stripped.startswith("ГЛАВА "):
			rest = stripped.split(maxsplit=1)[-1]
			if rest.isdigit():
				return True
			upper_rest = rest.upper()
			if upper_rest and all(ch in roman for ch in upper_rest):
				return True
		if " " in stripped:
			has_alpha = False
			all_upper = True
			for ch in stripped:
				if ch.isalpha():
					has_alpha = True
					if not ch.isupper():
						all_upper = False
				elif ch not in {" ", "\t"}:
					all_upper = False
			if has_alpha and all_upper:
				return True
	return False


def _starts_with_lowercase(text: str) -> bool:
	stripped = text.lstrip()
	if not stripped:
		return False
	if _HAS_REGEX:
		match = re.search(r"\p{L}", stripped)
		if not match:
			return False
		return re.match(r"\p{Ll}", match.group(0)) is not None
	for ch in stripped:
		if ch.isalpha():
			return ch.islower()
	return False


def clean_text(text: str, eos_token: str) -> str:
	text = text.replace("\r\n", "\n").replace("\r", "\n")
	text = text.replace("\f", "\n")
	text = _remove_control_chars(text)
	text = text.replace("*", "")

	lines = text.split("\n")
	kept_lines = []
	for line in lines:
		line = line.rstrip()
		if _is_page_number_line(line) or _is_chapter_line(line):
			continue
		kept_lines.append(line)
	text = "\n".join(kept_lines)

	if _HAS_REGEX:
		text = re.sub(r"(\p{L})-\n(\p{L})", r"\1\2", text)
	else:
		text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

	text = re.sub(r"\n{3,}", "\n\n", text)
	text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
	text = _normalize_inline_whitespace(text)
	text = re.sub(r" \n", "\n", text)
	text = re.sub(r"\n ", "\n", text)

	paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
	if not paragraphs:
		return ""
	merged = []
	for para in paragraphs:
		if merged and _starts_with_lowercase(para):
			merged[-1] = merged[-1].rstrip() + " " + para.lstrip()
		else:
			merged.append(para)
	return f" {eos_token}\n".join(merged)


def _starts_with_letter(text: str) -> bool:
	stripped = text.lstrip()
	if not stripped:
		return False
	if _HAS_REGEX:
		return re.search(r"\p{L}", stripped) is not None
	for ch in stripped:
		if ch.isalpha():
			return True
	return False


def _clean_line_for_streaming(line: str) -> str:
	line = _remove_control_chars(line)
	line = line.replace("*", "")
	line = line.rstrip()
	line = _normalize_inline_whitespace(line)
	return line


def _flush_paragraph(paragraph_lines: list[str]) -> str:
	if not paragraph_lines:
		return ""
	parts: list[str] = []
	for line in paragraph_lines:
		if not line:
			continue
		if parts and parts[-1].endswith("-") and _starts_with_letter(line):
			parts[-1] = parts[-1][:-1] + line.lstrip()
		else:
			parts.append(line)
	paragraph = " ".join(parts)
	paragraph = _normalize_inline_whitespace(paragraph)
	return paragraph.strip()


def clean_file_in_phases(input_path: Path, output_path: Path, eos_token: str) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)

	pending_paragraph: str | None = None
	current_lines: list[str] = []
	written_any = False

	def emit_paragraph(out_file, paragraph: str) -> tuple[str | None, bool]:
		nonlocal pending_paragraph, written_any
		if not paragraph:
			return pending_paragraph, written_any
		if pending_paragraph is None:
			pending_paragraph = paragraph
			return pending_paragraph, written_any
		if _starts_with_lowercase(paragraph):
			pending_paragraph = pending_paragraph.rstrip() + " " + paragraph.lstrip()
			return pending_paragraph, written_any
		if written_any:
			out_file.write(f" {eos_token}\n")
		out_file.write(pending_paragraph)
		written_any = True
		pending_paragraph = paragraph
		return pending_paragraph, written_any

	with input_path.open("r", encoding="utf-8", newline=None) as in_file, output_path.open("w", encoding="utf-8") as out_file:
		for raw_line in in_file:
			raw_line = raw_line.replace("\f", "\n")
			for segment in raw_line.split("\n"):
				line = _clean_line_for_streaming(segment)
				if _is_page_number_line(line) or _is_chapter_line(line):
					continue

				if not line.strip():
					paragraph = _flush_paragraph(current_lines)
					if paragraph:
						emit_paragraph(out_file, paragraph)
					current_lines = []
					continue

				current_lines.append(line)

		last_paragraph = _flush_paragraph(current_lines)
		if last_paragraph:
			emit_paragraph(out_file, last_paragraph)

		if pending_paragraph:
			if written_any:
				out_file.write(f" {eos_token}\n")
			out_file.write(pending_paragraph)

		out_file.write("\n")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Clean a book .txt for Transformer training.",
	)
	parser.add_argument("input", help="Input .txt file")
	parser.add_argument(
		"-o",
		"--output",
		default=None,
		help="Output .txt file (default: <input>-clean.txt)",
	)
	parser.add_argument(
		"--eos-token",
		default="</s>",
		help="Token inserted between paragraphs.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	input_path = Path(args.input)
	output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}-clean.txt")

	clean_file_in_phases(input_path, output_path, eos_token=args.eos_token)
	print(f"Wrote cleaned text to {output_path}")


if __name__ == "__main__":
	main()
