import os
from argparse import ArgumentParser
from typing import List

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor

from GlobalDataset import GlobalDataset
from bounding_box import BoundingBox
from mask_detector.utils.nms import single_class_non_max_suppression
from model import load_model
from util import model_to_device, get_bbox_lists, plot_image, bboxes_to_nms_input, get_clips, collate_fn, get_device


# noinspection PyShadowingNames
def evaluate(model, image_tensor_list: List[torch.Tensor], conf_thresh=0.5, iou_thresh=0.4) -> List[BoundingBox]:
    predictions = model(image_tensor_list)

    bbox_lists = get_bbox_lists(predictions, get_clips(image_tensor_list))

    return [np.array(bboxes)[single_class_non_max_suppression(
        *bboxes_to_nms_input(bboxes),
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh
    )] for i, bboxes in enumerate(bbox_lists)]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--session-id", type=str, required=False)
    parser.add_argument("--image-path", type=str, required=False)
    parser.add_argument("--batch-size", type=int, default=4)

    args = parser.parse_args()

    if not args.session_id:
        ids = [int(c.split("_")[1].split(".")[0]) for c in os.listdir("./checkpoints")]
        ids.sort(reverse=True)
        session_id = ids[0]
    else:
        session_id = args.session_id

    model = load_model(session_id)
    model.eval()
    model_to_device(model)

    if args.image_path is not None:
        image = Image.open(args.image_path).convert("RGB")
        image_tensor = ToTensor()(image).to(get_device())

        accepted_bboxes = evaluate(model, [image_tensor])[0]
        plot_image(image_tensor, accepted_bboxes)
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset=GlobalDataset(transforms=Compose([ToTensor()])),
            batch_size=args.batch_size,
            collate_fn=collate_fn
        )
        for image_tensor, _ in data_loader:
            accepted_bbox_lists = evaluate(model, image_tensor)
            for i, accepted_bboxes in enumerate(accepted_bbox_lists):
                plot_image(image_tensor[i], accepted_bboxes)
            break
