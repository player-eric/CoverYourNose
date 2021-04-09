import torch
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from os import getcwd, listdir, path, mkdir
from argparse import ArgumentParser
import re
from ntpath import basename

"""
SSD300 Source ~ https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Detection/SSD/src/model.py
SSD300 Repo ~ https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD#model-overview
Tutorial ~ https://pytorch.org/hub/nvidia_deeplearningexamples_ssd/
"""

parser = ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument("--data-dir", default="./data")
parser.add_argument("--absolute-image-uris", nargs="+")
args = parser.parse_args()

acceptable_image_suffixes = [".png", ".jpg", ".jpeg"]
image_filter = rf"^.*({'|'.join(acceptable_image_suffixes)})$"

if not path.isdir("./output"):
    mkdir("./output")

if args.data_dir:
    uris = [path.join(args.data_dir, i) for i in listdir(args.data_dir)]
else:
    uris = args.absolute_image_uris

uris = list(filter(lambda i: re.match(image_filter, i), uris))
if not len(uris):
    print(f"{args.data_dir} did not contain any files ending in ({'|'.join(acceptable_image_suffixes)})")
    exit()

precision = "fp32"
confidence_threshold = 0.40

ssd_model = torch.hub.load(
    'NVIDIA/DeepLearningExamples:torchhub',
    'nvidia_ssd',
    model_math=precision,
    map_location=torch.device('cpu')
)
ssd_model.to('cuda')
ssd_model.eval()

utils = torch.hub.load(
    'NVIDIA/DeepLearningExamples:torchhub',
    'nvidia_ssd_processing_utils'
)

inputs = [utils.prepare_input(uri) for uri in uris]
tensor = utils.prepare_tensor(inputs, precision == "fp16")

print(f"\nInput tensor shape: {tensor.shape}\n")

with torch.no_grad():
    detections_batch = ssd_model(tensor)

for i, detection in enumerate(detections_batch):
    print(f"Detection tensor {i + 1} shape: {detection.shape}")

print("\nProcessing...\n")

results_per_input = utils.decode_results(detections_batch)
best_results_per_input = [utils.pick_best(results, confidence_threshold) for results in results_per_input]

classes_to_labels = utils.get_coco_object_dictionary()

print("Plotting...")

for image_idx in range(len(best_results_per_input)):
    fig, ax = plt.subplots(1)
    # Show original, denormalized image...
    image = inputs[image_idx] / 2 + 0.5
    ax.imshow(image)
    # ...with detections
    bboxes, classes, confidences = best_results_per_input[image_idx]

    print(f"\n__ Image {image_idx + 1} __")
    for i, [bbox, classification, confidence] in enumerate(list(zip(bboxes, classes, confidences))):
        print(f"\n{i + 1}) Bounding box: {bbox}")
        print(f"{i + 1}) Class: {classes_to_labels[classification]}")
        print(f"{i + 1}) Confidence: {confidence}")

    for idx in range(len(bboxes)):
        left, bot, right, top = bboxes[idx]
        x, y, w, h = [val * 300 for val in [left, bot, right - left, top - bot]]
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y, "{} {:.0f}%".format(classes_to_labels[classes[idx] - 1], confidences[idx]*100),
                bbox=dict(facecolor='white', alpha=0.5))

    output_path = f"{getcwd()}/output/{basename(uris[image_idx])}"
    plt.savefig(output_path)
    plt.clf()

    print(f"\nSaved to {output_path}")
