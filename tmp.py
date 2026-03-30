import sentencepiece as spm
import os


sp = spm.SentencePieceProcessor()
sp.load("tokenizer/andric_sp.model")

with open("proba.txt", 'r', encoding='utf-8') as infile:
    for line in infile:
        ids = sp.encode(line, out_type=str)
        print(ids)
        ids = sp.encode(line, out_type=int)
        print(ids)

        print(sp.decode(ids))
        break
    




