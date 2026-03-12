import os
import pandas as pd

base_path = "runs/train"

folds = [
    "fly_fold_1",
    "fly_fold_2",
    "fly_fold_3",
    "fly_fold_4",
    "fly_fold_5"
]

map50_list = []
map5095_list = []
precision_list = []
recall_list = []

for fold in folds:
    csv_path = os.path.join(base_path, fold, "results.csv")
    df = pd.read_csv(csv_path)

    # 🔥 HAPUS SPASI TERSEMBUNYI
    df.columns = df.columns.str.strip()

    best_row = df.iloc[-1]

    map50_list.append(best_row["metrics/mAP_0.5"])
    map5095_list.append(best_row["metrics/mAP_0.5:0.95"])
    precision_list.append(best_row["metrics/precision"])
    recall_list.append(best_row["metrics/recall"])

print("\n===== HASIL 5-FOLD CROSS VALIDATION =====\n")

print("mAP@0.5 per fold:", map50_list)
print("mAP@0.5:0.95 per fold:", map5095_list)

print("\n===== RATA-RATA =====\n")
print("Average mAP@0.5:", sum(map50_list)/5)
print("Average mAP@0.5:0.95:", sum(map5095_list)/5)
print("Average Precision:", sum(precision_list)/5)
print("Average Recall:", sum(recall_list)/5)