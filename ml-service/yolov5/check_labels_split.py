import os
from collections import Counter

LABEL_DIRS = [
    "labelImg_SPLIT/labels/train",
    "labelImg_SPLIT/labels/val"
]

counter = Counter()

for label_dir in LABEL_DIRS:
    for file in os.listdir(label_dir):
        if file.endswith(".txt"):
            with open(os.path.join(label_dir, file)) as f:
                for line in f:
                    cls = int(float(line.strip().split()[0]))
                    counter[cls] += 1

print("Distribusi label labelImg_SPLIT:")
for k, v in sorted(counter.items()):
    print(f"Class {k}: {v} instances")
