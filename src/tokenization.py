import sentencepiece as spm


def encode_with_inline_eos(text: str, sp: spm.SentencePieceProcessor, marker: str = "</s>") -> list[int]:
	if not marker:
		return sp.encode(text, out_type=int)

	parts = text.split(marker)
	if len(parts) == 1:
		return sp.encode(text, out_type=int)

	eos_id = sp.eos_id()
	if eos_id < 0:
		raise ValueError("Tokenizer has no EOS id; cannot map inline </s> markers.")

	ids: list[int] = []
	for i, part in enumerate(parts):
		if part:
			ids.extend(sp.encode(part, out_type=int, add_bos=False, add_eos=False))
		if i < len(parts) - 1:
			ids.append(eos_id)

	return ids