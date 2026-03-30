import sys
import math

def main():
    if len(sys.argv) != 3:
        print("Upotreba: python uzmi_procenat.py <procenat> <y>")
        sys.exit(1)

    try:
        procenat = int(sys.argv[1])
        y = int(sys.argv[2])
        if procenat <= 0 or procenat > 100:
            raise ValueError("Procenat mora biti izmedju 1 i 100.")
        if y < 1:
            raise ValueError("Drugi argument 'y' mora biti 1 ili veci.")
    except ValueError as e:
        print(f"Greska: {e}")
        print("Molim vas unesite validan celobrojni procenat (1-100) i segment 'y' (broj >= 1).")
        sys.exit(1)

    input_fajl = "sveKnjige.txt"
    output_fajl = f"sveKnjige{procenat}-{y}.txt"

    ukupan_broj_linija = 29545211

    broj_linija_za_uzimanje = math.ceil(ukupan_broj_linija * (procenat / 100.0))
    pocetna_linija = (y - 1) * broj_linija_za_uzimanje
    krajnja_linija = min(y * broj_linija_za_uzimanje, ukupan_broj_linija)

    if pocetna_linija >= ukupan_broj_linija:
        print(f"Ovaj segment preskace sve linije (pocetna linija {pocetna_linija} je veca od ukupnog broja {ukupan_broj_linija}).")
        sys.exit(0)

    print(f"Ukupan broj linija: {ukupan_broj_linija}")
    print(f"Uzimam segment {y} (po {procenat}%): od linije {pocetna_linija} do {krajnja_linija-1}...")

    with open(input_fajl, 'r', encoding='utf-8') as fin, \
         open(output_fajl, 'w', encoding='utf-8') as fout:
        for i, linija in enumerate(fin):
            if i < pocetna_linija:
                continue
            if i >= krajnja_linija:
                break
            fout.write(linija)

    print(f"Uspesno sacuvano u {output_fajl}")

if __name__ == "__main__":
    main()
