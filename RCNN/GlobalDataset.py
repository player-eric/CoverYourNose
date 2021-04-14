import os

from PIL import Image

from util import generate_target


class GlobalDataset(object):

    def __init__(self, transforms):
        self.transforms = transforms
        self.image_dirs = [
            # "../Data/AIZOO/train/images",
            "../Data/Kaggle1/images",
        ]
        self.annotation_dirs = [
            # "../Data/AIZOO/train/annotations",
            "../Data/Kaggle1/annotations",
        ]
        self.classes = [
            # {
            #     "face": 0,
            #     "face_mask": 1,
            # },
            {
                "without_mask": 0,
                "with_mask": 1,
                "mask_weared_incorrect": 2,
            },
        ]
        # load all image files, sorting them to
        # ensure that they are aligned
        self.index_thresholds = []
        self.image_names = []
        self.annotation_names = []

        for image_dir, annotation_dir in zip(self.image_dirs, self.annotation_dirs):
            image_names = list(sorted(os.listdir(image_dir)))
            annotation_names = list(sorted(os.listdir(annotation_dir)))

            assert len(image_names) == len(annotation_names)

            self.image_names.extend(image_names)
            self.annotation_names.extend(annotation_names)

            acc = 0 if not len(self.index_thresholds) else self.index_thresholds[-1]
            self.index_thresholds.append(acc + len(image_names))

    def __getitem__(self, idx):
        i = next(x for x, t in enumerate(self.index_thresholds) if idx < t)

        image_file = self.image_names[idx]
        annotation_file = self.annotation_names[idx]
        img_path = os.path.join(self.image_dirs[i], image_file)
        annotation_path = os.path.join(self.annotation_dirs[i], annotation_file)

        # Generate Label
        target = generate_target(idx, annotation_path, self.classes[i])

        img = Image.open(img_path).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.image_names)
