import os
import re

def clean_text(text):
    text = text.strip()
    # Basic cleanup if necessary (e.g., removing multiple spaces)
    text = re.sub(r'\s+', ' ', text)
    return text

def is_good_sequence(text):
    # Filter out empty or very short sequences
    if len(text) < 20 or text[-1] != ".":
        return False
    # Filter out sequences with too few words
    words = text.split()
    if len(words) < 4:
        return False
    # Heuristic: must contain some alphabetic characters to avoid purely numeric/symbolic junk
    if not re.search(r'[A-Za-zА-Яа-яЂЈЋЏЉЊђјћџљњ]', text):
        return False
    return True

def prepare_data(input_path, output_path):
    print(f"Reading from {input_path}")
    
    total_lines = 0
    kept_lines = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            total_lines += 1
            cleaned = clean_text(line)
            
            if is_good_sequence(cleaned):
                final_sequence = f"{cleaned} \n"
                outfile.write(final_sequence)
                kept_lines += 1
                
            if total_lines % 100000 == 0:
                print(f"Processed {total_lines} lines...")

    print(f"Done! Original lines: {total_lines}, Kept lines: {kept_lines}")
    print(f"Output saved to {output_path}")

if __name__ == '__main__':
    input_file = os.path.join(os.path.dirname(__file__), 'srWaC', 'srWaC-cirilica.txt')
    output_file = os.path.join(os.path.dirname(__file__), 'srWaC', 'srWaC-clean.txt')
    prepare_data(input_file, output_file)
