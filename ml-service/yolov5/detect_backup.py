# YOLOv5 Detect + Object Counting
# Author: Custom R&D Version

import argparse
import csv
import os
from pathlib import Path

import cv2
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in os.sys.path:
    os.sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (
    check_img_size,
    non_max_suppression,
    scale_boxes,
    increment_path,
)
from utils.torch_utils import select_device, smart_inference_mode
from ultralytics.utils.plotting import Annotator, colors


@smart_inference_mode()
def run(
    weights,
    source,
    data,
    imgsz=640,
    conf_thres=0.4,
    iou_thres=0.45,
    device="",
    save_csv=True,
    project=ROOT / "runs/detect",
    name="exp",
):

    # ================= SETUP =================
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, data=data)
    stride, names = model.stride, model.names
    imgsz = check_img_size(imgsz, s=stride)

    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=True)

    save_dir = increment_path(Path(project) / name)
    save_dir.mkdir(parents=True, exist_ok=True)

    csv_path = save_dir / "count_results.csv"
    if save_csv:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "class", "count"])

    # ================= INFERENCE =================
    for path, im, im0, _, _ in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.float() / 255.0
        if len(im.shape) == 3:
            im = im.unsqueeze(0)

        pred = model(im)
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        annotator = Annotator(im0, line_width=2)
        class_counts = {}

        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in det:
                    cls_id = int(cls)
                    cls_name = names[cls_id]

                    # count
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

                    # draw box
                    label = f"{cls_name} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(cls_id, True))

        # ================= DISPLAY COUNT =================
        y = 30
        for cls_name, count in class_counts.items():
            text = f"{cls_name}: {count}"
            cv2.putText(
                im0,
                text,
                (25, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,4
                (0, 255, 255),
                3,
                cv2.LINE_AA,
            )
            y += 35

            if save_csv:
                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([Path(path).name, cls_name, count])

        # ================= SAVE IMAGE =================
        save_path = save_dir / Path(path).name
        cv2.imwrite(str(save_path), annotator.result())

    print(f"\n✅ Results saved in: {save_dir}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="model path")
    parser.add_argument("--source", type=str, required=True, help="image / folder")
    parser.add_argument("--data", type=str, required=True, help="dataset yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--device", default="")
    parser.add_argument("--name", default="exp")
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    run(
        weights=opt.weights,
        source=opt.source,
        data=opt.data,
        imgsz=opt.imgsz,
        conf_thres=opt.conf,
        name=opt.name,
    )
