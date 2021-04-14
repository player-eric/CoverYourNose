import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from bs4 import BeautifulSoup

from bounding_box import bbox_from_two_points

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Device: {device}")


def generate_target(image_id, file, classes):
    with open(file) as f:
        data = f.read()
        soup = BeautifulSoup(data, 'lxml')
        objects = soup.find_all('object')

        boxes = []
        labels = []
        for i in objects:
            boxes.append(generate_box(i))
            labels.append(generate_label(i, classes))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Labels (In my case, I only one class: target class or background)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([image_id])
        # Annotation is in dictionary format
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


def get_bboxes(dictionaries, clip):
    all_bboxes = []

    for dict in dictionaries:
        boxes = dict["boxes"].detach().cpu().numpy()
        labels = dict["labels"].detach().cpu().numpy()
        confidences = dict["scores"].detach().cpu().numpy()

        bboxes = []
        for [xmin, ymin, xmax, ymax], label, confidence in zip(boxes, labels, confidences):
            bbox = bbox_from_two_points("face_mask_roi", xmin, ymin, xmax, ymax, clip)
            bbox.set("predicted_class", label)
            bbox.set("confidence", confidence)
            bboxes.append(bbox)

        all_bboxes.append(bboxes)

    return np.array(all_bboxes)


def bboxes_to_nms_input(bboxes):
    bboxes_numpy = []
    confidences_numpy = []

    for bbox in bboxes:
        bboxes_numpy.append(bbox.points)
        confidences_numpy.append(bbox.get("confidence"))

    return np.stack(bboxes_numpy), np.array(confidences_numpy)


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
        session_id, is_prediction = save_info
        plt.savefig(f"./output/{'prediction' if is_prediction else 'true'}_{session_id}.png")

    plt.show()
