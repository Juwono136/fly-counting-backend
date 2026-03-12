import os
from collections import Counter

LABEL_DIRS = [
    "Dataset_Roboflow/train/labels",
    "Dataset_Roboflow/valid/labels",
    "Dataset_Roboflow/test/labels",
]

counter = Counter()

for label_dir in LABEL_DIRS:
    for file in os.listdir(label_dir):
        if file.endswith(".txt"):
            with open(os.path.join(label_dir, file)) as f:
                for line in f:
                    cls = int(float(line.strip().split()[0]))
                    counter[cls] += 1

print("Distribusi label Dataset_Roboflow:")
for k, v in sorted(counter.items()):
    print(f"Class {k}: {v} instances")
