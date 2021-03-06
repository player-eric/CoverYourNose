import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from bs4 import BeautifulSoup

from bounding_box import bbox_from_two_points

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Device: {device}")


def get_device():
    return device


def generate_target(image_id, file, classes):
    with open(file) as f:
        data = f.read()
        soup = BeautifulSoup(data, 'lxml')
        objects = soup.find_all('object')

        if len(objects):
            boxes = []
            labels = []
            for i in objects:
                boxes.append(generate_box(i))
                labels.append(generate_label(i, classes))
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.empty(0, 4, dtype=torch.float32)
            labels = torch.empty(0, dtype=torch.int64)

        # Tensorise img_id
        img_id = torch.tensor([image_id])
        # Output annotation in dictionary format
        return {"boxes": boxes, "labels": labels, "image_id": img_id}


def generate_box(obj):
    x1 = int(obj.find('xmin').text)
    y1 = int(obj.find('ymin').text)
    x2 = int(obj.find('xmax').text)
    y2 = int(obj.find('ymax').text)

    return [x1, y1, x2, y2]


def generate_label(obj, classes):
    return classes[obj.find('name').text]


def input_to_device(imgs, annotations):
    imgs = list(img.to(device) for img in imgs)
    annotations = [{k: v.to(device) for k, v in a.items()} for a in annotations]
    return imgs, annotations


def model_to_device(model):
    model.to(device)


def get_clips(image_tensors):
    return np.array([tuple(t.shape[1:][::-1]) for t in image_tensors])


def get_bbox_lists(dictionaries, clips):
    all_bboxes = []

    for dict, clip in zip(dictionaries, clips):
        boxes = dict["boxes"].detach().cpu().numpy()
        labels = dict["labels"].detach().cpu().numpy()
        if "scores" in dict:
            confidences = dict["scores"].detach().cpu().numpy()
        else:
            confidences = [1] * len(boxes)

        bboxes = []
        for [x1, y1, x2, y2], label, confidence in zip(boxes, labels, confidences):
            bbox = bbox_from_two_points("face_mask_roi", x1, y1, x2, y2, clip)
            bbox.set("predicted_class", label)
            bbox.set("confidence", confidence)
            bboxes.append(bbox)

        all_bboxes.append(bboxes)

    return all_bboxes


def bboxes_to_nms_input(bboxes):
    bboxes_numpy = []
    confidences_numpy = []

    for bbox in bboxes:
        bboxes_numpy.append(bbox.points)
        confidences_numpy.append(bbox.get("confidence"))

    bboxes_numpy = np.stack(bboxes_numpy) if len(bboxes_numpy) else np.array(bboxes_numpy)
    confidences_numpy = np.array(confidences_numpy)

    return bboxes_numpy, confidences_numpy


def plot_image(image, bboxes, save_info=None):
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(image.permute(1, 2, 0))

    for box in bboxes:
        # Create a Rectangle patch
        rect = patches.Rectangle(
            box.top_left,
            box.width,
            box.height,
            linewidth=2,
            edgecolor=["r", "g", "b"][box.get("predicted_class")],
            facecolor="none"
        )

        # Add the patch to the Axes
        ax.add_patch(rect)

    if save_info is not None:
        i, session_id, is_prediction = save_info
        plt.savefig(f"./output/{'prediction' if is_prediction else 'true'}_{session_id}_{i}.png")

    plt.show()


def collate_fn(batch):
    return tuple(zip(*batch))
