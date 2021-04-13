from bs4 import BeautifulSoup
import torch
import os
from PIL import Image


class Kaggle1Dataset(object):

    def __init__(self, transforms):
        self.transforms = transforms
        self.image_dir = "../Data/Kaggle1/images"
        self.annotation_dir = "../Data/Kaggle1/annotations"
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(self.image_dir)))
        self.labels = list(sorted(os.listdir(self.annotation_dir)))
        self.classes = {
            "without_mask": 0,
            "with_mask": 1,
            "mask_weared_incorrect": 2,
        }

    def __getitem__(self, idx):
        # load images ad masks
        file_image = 'maksssksksss' + str(idx) + '.png'
        file_label = 'maksssksksss' + str(idx) + '.xml'
        img_path = os.path.join(self.image_dir, file_image)
        label_path = os.path.join(self.annotation_dir, file_label)
        img = Image.open(img_path).convert("RGB")
        # Generate Label
        target = self.generate_target(idx, label_path)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

    def generate_target(self, image_id, file):
        with open(file) as f:
            data = f.read()
            soup = BeautifulSoup(data, 'xml')
            objects = soup.find_all('object')

            # Bounding boxes for objects
            # In coco format, bbox = [xmin, ymin, width, height]
            # In pytorch, the input should be [xmin, ymin, xmax, ymax]
            boxes = []
            labels = []
            for i in objects:
                boxes.append(self.generate_box(i))
                labels.append(self.generate_label(i))

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # Labels (In my case, I only one class: target class or background)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            # Tensorise img_id
            img_id = torch.tensor([image_id])
            # Annotation is in dictionary format
            return {"boxes": boxes, "labels": labels, "image_id": img_id}

    @staticmethod
    def generate_box(obj):
        x_min = int(obj.find('xmin').text)
        y_min = int(obj.find('ymin').text)
        x_max = int(obj.find('xmax').text)
        y_max = int(obj.find('ymax').text)

        return [x_min, y_min, x_max, y_max]

    def generate_label(self, obj):
        return self.classes[obj.find('name').text]

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))
