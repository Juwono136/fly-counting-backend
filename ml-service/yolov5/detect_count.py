import argparse
import os
from pathlib import Path
import torch
import cv2
from collections import defaultdict

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (
    check_img_size,
    non_max_suppression,
    scale_boxes,
    increment_path,
)
from utils.torch_utils import select_device
from ultralytics.utils.plotting import Annotator, colors

# =============================
# MAIN FUNCTION
# =============================
@torch.no_grad()
def run(
    weights,
    source,
    data,
    imgsz=640,
    conf_thres=0.4,
    iou_thres=0.45,
    device="",
    project="runs/detect",
    name="exp",
    line_thickness=3,
):

    save_dir = increment_path(Path(project) / name)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, data=data)
    stride, names = model.stride, model.names
    imgsz = check_img_size(imgsz, s=stride)

    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    model.warmup(imgsz=(1, 3, imgsz, imgsz))

    for path, im, im0, _, _ in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.float() / 255.0
        if im.ndimension() == 3:
            im = im.unsqueeze(0)

        pred = model(im)
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        annotator = Annotator(im0, line_width=line_thickness)
        class_counter = defaultdict(int)

        if len(pred[0]):
            pred[0][:, :4] = scale_boxes(im.shape[2:], pred[0][:, :4], im0.shape).round()

            for *xyxy, conf, cls in pred[0]:
                cls = int(cls)
                class_counter[names[cls]] += 1
                label = f"{names[cls]} {conf:.2f}"
                annotator.box_label(xyxy, label, color=colors(cls, True))

        # =============================
        # DRAW COUNTING TEXT (BIG & CLEAR)
        # =============================
        y_offset = 40
        for cls_name, count in class_counter.items():
            text = f"{cls_name}: {count}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.6      # 🔥 BESAR
            thickness = 3

            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

            # background box
            cv2.rectangle(
                im0,
                (15, y_offset - th - 12),
                (15 + tw + 12, y_offset),
                (0, 0, 0),
                -1,
            )

            # text
            cv2.putText(
                im0,
                text,
                (20, y_offset - 5),
                font,
                font_scale,
                (0, 255, 255),  # kuning terang
                thickness,
                cv2.LINE_AA,
            )

            y_offset += 45

        # save image
        save_path = save_dir / Path(path).name
        cv2.imwrite(str(save_path), im0)

    print(f"✅ Results saved in: {save_dir}")


# =============================
# ARGUMENT PARSER
# =============================
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="exp")
    parser.add_argument("--line-thickness", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    run(
        weights=opt.weights,
        source=opt.source,
        data=opt.data,
        imgsz=opt.imgsz,
        conf_thres=opt.conf,
        iou_thres=opt.iou,
        device=opt.device,
        project=opt.project,
        name=opt.name,
        line_thickness=opt.line_thickness,
    )
