import os

LABEL_DIR = "../Dataset_Lalat/labels"
MAX_CLASS_ID = 4   # kelas 0–4

error_found = False

for root, _, files in os.walk(LABEL_DIR):
    for f in files:
        if f.endswith(".txt"):
            path = os.path.join(root, f)
            with open(path) as file:
                for i, line in enumerate(file, 1):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        print(f"[FORMAT ERROR] {path} line {i}: {line.strip()}")
                        error_found = True
                    elif not parts[0].isdigit() or int(parts[0]) > MAX_CLASS_ID:
                        print(f"[CLASS ERROR] {path} line {i}: {line.strip()}")
                        error_found = True

if not error_found:
    print("✅ Semua label valid. Dataset bersih.")
