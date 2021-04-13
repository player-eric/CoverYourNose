import torch
from torchvision import transforms

from RCNN.Kaggle1Dataset import Kaggle1Dataset
from RCNN.model import get_model_instance_segmentation, save_model
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def train(num_epochs):
    model.to(device)

    optimizer = torch.optim.SGD(
        params=[p for p in model.parameters() if p.requires_grad],
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )

    for epoch in range(num_epochs):
        model.train()

        epoch_loss = 0
        for t_imgs, t_annotations in data_loader:
            t_imgs = list(img.to(device) for img in t_imgs)
            t_annotations = [{k: v.to(device) for k, v in t.items()} for t in t_annotations]
            loss_dict = model([t_imgs[0]], [t_annotations[0]])
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            epoch_loss += losses

        print(epoch_loss)


def plot_image(img_tensor, annotation):
    fig, ax = plt.subplots(1)
    img = img_tensor.cpu().data

    # Display the image
    ax.imshow(img.permute(1, 2, 0))

    for box in annotation["boxes"]:
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

    plt.show()


if __name__ == "__main__":
    dataset = Kaggle1Dataset(
        transforms=transforms.Compose([transforms.ToTensor()])
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        collate_fn=dataset.collate_fn
    )

    model = get_model_instance_segmentation(len(dataset.classes))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train(num_epochs=25)

    for imgs, annotations in data_loader:
        model.eval()

        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]

        predictions = model(imgs)

        print("Prediction")
        plot_image(imgs[2], predictions[2])

        print("Target")
        plot_image(imgs[2], annotations[2])

        break

    save_model(model)
