import os

from PIL import Image

from util import generate_target


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
        target = generate_target(idx, label_path, self.classes)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))
