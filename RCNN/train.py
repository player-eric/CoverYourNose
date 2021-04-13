import os
from argparse import ArgumentParser

import torch
from torchvision import transforms

from Kaggle1Dataset import Kaggle1Dataset
from util import input_to_device, plot_image, model_to_device
from model import get_model_instance_segmentation, save_model


def train(num_epochs):
    model_to_device(model)

    params = [p for p in model.parameters()]
    trainable = [p for p in params if p.requires_grad]
    print(f"{len(params)} of {len(trainable)} are trainable.")

    optimizer = torch.optim.SGD(
        params=trainable,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )

    for epoch in range(num_epochs):
        print(f"Beginning epoch {epoch + 1} of {num_epochs}.")

        model.train()

        epoch_loss = 0
        for t_imgs, t_annotations in data_loader:
            t_imgs, t_annotations = input_to_device(t_imgs, t_annotations)

            loss_dict = model([t_imgs[0]], [t_annotations[0]])
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            epoch_loss += losses

        print(epoch_loss)

        save_model(model)

        print("Saved checkpoint, in place.")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--num-epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--single-eval", action="store_true", default=False)

    args = parser.parse_args()

    print(args)

    if not os.path.isdir("./output"):
        os.mkdir("./output")
    if not os.path.isdir("./checkpoints"):
        os.mkdir("./checkpoints")

    print("Instantiating dataset...", end="")
    dataset = Kaggle1Dataset(
        transforms=transforms.Compose([transforms.ToTensor()])
    )
    print("Done.")

    print("Instantiating data loader...", end="")
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=dataset.collate_fn
    )
    print("Done.")

    print("Instantiating model...", end="")
    model = get_model_instance_segmentation()
    print("Done.")

    print("Beginning training...")
    train(num_epochs=args.num_epochs)
    print("Training done.")

    if args.single_eval:
        print("Beginning evaluation...")
        for imgs, annotations in data_loader:
            model.eval()

            imgs, annotations = input_to_device(imgs, annotations)

            predictions = model(imgs)

            print("Saving prediction image...", end="")
            plot_image(imgs[2], predictions[2], prediction=True, save=True)
            print("Done.")

            print("Saving ground truth image...", end="")
            plot_image(imgs[2], annotations[2], prediction=False, save=True)
            print("Done.")

            break
        print("Evaluation done.")

    print("Saving model...", end="")
    save_model(model)
    print("Done.")
