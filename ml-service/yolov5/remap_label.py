from pathlib import Path

label_dirs = [
    Path("Dataset_Lalat_2class/labels/train"),
    Path("Dataset_Lalat_2class/labels/val")
]

for label_dir in label_dirs:
    for label_file in label_dir.glob("*.txt"):
        lines = label_file.read_text().strip().splitlines()
        new_lines = []

        for line in lines:
            parts = line.split()

            if parts[0] == "0":          # agas
                new_lines.append(" ".join(parts))

            elif parts[0] == "3":        # lalat hijau
                parts[0] = "1"
                new_lines.append(" ".join(parts))

        label_file.write_text("\n".join(new_lines))

print("✅ Remap & filter selesai: 0=agas, 1=lalat_hijau")

