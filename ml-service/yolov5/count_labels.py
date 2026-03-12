import os
from collections import Counter

label_dir = r"Dataset_Lalat_2class_aug/labels/train"

counter = Counter()

for file in os.listdir(label_dir):
    if file.endswith(".txt"):
        with open(os.path.join(label_dir, file), "r") as f:
            for line in f:
                cls = int(float(line.strip().split()[0]))
                counter[cls] += 1

print("Distribusi label:")
for k, v in counter.items():
    print(f"Class {k}: {v} instances")
