import os
from argparse import ArgumentParser

import torch
from PIL import Image
from torchvision.transforms import ToTensor

from mask_detector.utils.nms import single_class_non_max_suppression
from model import load_model
from util import model_to_device, get_bboxes, plot_image, bboxes_to_nms_input

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--session-id", type=str, required=False)
    parser.add_argument("--image-path", type=str, required=True)
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

    pil_image = Image.open(args.image_path).convert("RGB")
    image_tensor = ToTensor()(pil_image)

    predictions = model(torch.unsqueeze(image_tensor, 0))

    bboxes = get_bboxes(predictions, pil_image.size)[0]
    bboxes_numpy, confidences_numpy = bboxes_to_nms_input(bboxes)

    accepted_bboxes = bboxes[single_class_non_max_suppression(
        bboxes=bboxes_numpy,
        confidences=confidences_numpy,
        conf_thresh=0.5,
        iou_thresh=0.4
    )]

    plot_image(image_tensor, accepted_bboxes)
