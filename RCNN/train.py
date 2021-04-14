import math
import os
from argparse import ArgumentParser
from time import time

import torch
from torchvision.transforms import Compose, ToTensor, RandomAdjustSharpness, RandomAutocontrast

from GlobalDataset import GlobalDataset
from model import save_model, load_model
from util import input_to_device, model_to_device, collate_fn


def train(num_epochs):
    model_to_device(model)

    params = [p for p in model.parameters()]
    trainable = [p for p in params if p.requires_grad]
    print(f"{len(trainable)} of {len(params)} model parameters are trainable.")

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

            loss_dict = model(t_imgs, t_annotations)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            epoch_loss += losses

        print(epoch_loss)

        print(f"Overwriting ./checkpoints/rcnn_box_predictor_{session_id}.pt with newest weights...", end="")
        save_model(model, session_id)
        print("Done.")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--num-epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--session-id", type=str, required=False)

    args = parser.parse_args()

    print(args)

    print(f"Instantiating model (session_id = {args.session_id})...", end="")
    model = load_model(args.session_id)
    print("Done.")

    session_id = math.floor(time())
    print(f"Session ID: {session_id}")

    if not os.path.isdir("./output"):
        os.mkdir("./output")
    if not os.path.isdir("./checkpoints"):
        os.mkdir("./checkpoints")

    print("Instantiating data loader...", end="")
    data_loader = torch.utils.data.DataLoader(
        dataset=GlobalDataset(transforms=Compose([
            ToTensor(),
            RandomAutocontrast(0.1),
            RandomAdjustSharpness(0.8, 0.1)
        ])),
        batch_size=args.batch_size,
        collate_fn=collate_fn
    )
    print("Done.")

    print("Beginning training...")
    train(num_epochs=args.num_epochs)
    print("Training done.")
