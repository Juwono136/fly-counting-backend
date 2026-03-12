import os
from collections import Counter

PRED_DIR = "runs/detect/test_visualisasi3/labels"

counter = Counter()

for file in os.listdir(PRED_DIR):
    if file.endswith(".txt"):
        with open(os.path.join(PRED_DIR, file)) as f:
            for line in f:
                cls = int(float(line.strip().split()[0]))
                counter[cls] += 1

print("\n=== Total Hasil Deteksi ===")
for k, v in sorted(counter.items()):
    if k == 0:
        print(f"Agas : {v}")
    elif k == 1:
        print(f"Lalat Hijau : {v}")
