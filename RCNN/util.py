import math
from time import time

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Device: {device}")


def input_to_device(imgs, annotations):
    imgs = list(img.to(device) for img in imgs)
    annotations = [{k: v.to(device) for k, v in a.items()} for a in annotations]
    return imgs, annotations


def model_to_device(model):
    model.to(device)


def plot_image(img_tensor, annotation, save=True):
    fig, ax = plt.subplots(1)
    img = img_tensor.cpu().data

    # Display the image
    ax.imshow(img.permute(1, 2, 0))

    for box in annotation["boxes"].detach().cpu().numpy():
        xmin, ymin, xmax, ymax = box

        # Create a Rectangle patch
        rect = patches.Rectangle(
            (xmin, ymin),
            (xmax - xmin),
            (ymax - ymin),
            linewidth=1,
            edgecolor='r',
            facecolor='none'
        )

        # Add the patch to the Axes
        ax.add_patch(rect)

    if save:
        plt.savefig(f"./output/prediction_{math.floor(time())}.png")

    plt.show()
