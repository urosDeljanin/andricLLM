import argparse
import os
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import login

login(token=os.environ.get("HF_TOKEN"))

def _compact_to_single_line(text: str) -> str:
    # Ensure every dataset entry is represented as exactly one output line.
    return " ".join(text.split())


def export_oscar(
    dataset_name: str,
    config_name: str,
    split_name: str,
    output_path: Path,
    text_field: str | None,
    max_docs: int | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        dataset_name,
        config_name,
        split=split_name,
        streaming=True,
    )

    saved = 0
    skipped = 0
    detected_field = text_field

    with output_path.open("w", encoding="utf-8") as out_file:
        for record in dataset:
            if detected_field is None:
                if "text" in record:
                    detected_field = "text"
                elif "content" in record:
                    detected_field = "content"
                else:
                    raise KeyError(
                        "Nije pronadjeno tekstualno polje. Prosledi --text-field."
                    )

            value = record.get(detected_field)
            if not isinstance(value, str):
                skipped += 1
                continue

            line = _compact_to_single_line(value)
            if not line:
                skipped += 1
                continue

            out_file.write(line + "\n")
            saved += 1

            if saved % 100000 == 0:
                print(f"Sacuvano {saved} dokumenata...")

            if max_docs is not None and saved >= max_docs:
                break

    print(f"Gotovo. Sacuvano: {saved}, preskoceno: {skipped}")
    print(f"Izlazni fajl: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preuzmi OSCAR korpus i sacuvaj srpski tekst u jedan txt fajl (jedan unos = jedna linija)."
    )
    parser.add_argument(
        "--dataset",
        default="oscar-corpus/OSCAR-2301",
        help="Naziv dataseta na Hugging Face (podrazumevano: oscar-corpus/OSCAR-2301)",
    )
    parser.add_argument(
        "--config",
        default="sr",
        help="Dataset config/jezik (podrazumevano: sr)",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split (podrazumevano: train)",
    )
    parser.add_argument(
        "--text-field",
        default=None,
        help="Ime polja koje sadrzi tekst (npr. text ili content). Ako nije dato, detektuje se automatski.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Opcionalno ogranicenje broja dokumenata za test.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("oscar/oscar-sr.txt"),
        help="Putanja izlaznog fajla (podrazumevano: books/oscar/oscar-sr.txt)",
    )

    args = parser.parse_args()

    export_oscar(
        dataset_name=args.dataset,
        config_name=args.config,
        split_name=args.split,
        output_path=args.output,
        text_field=args.text_field,
        max_docs=args.max_docs,
    )


if __name__ == "__main__":
    main()