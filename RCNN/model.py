import torch
from torchvision.models.detection import faster_rcnn, fasterrcnn_resnet50_fpn


def get_model_instance_segmentation():
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, 3)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.roi_heads.box_predictor.parameters():
        param.requires_grad = True

    return model


def save_model(model, session_id):
    torch.save(model.roi_heads.box_predictor.state_dict(), f"./checkpoints/rcnn_box_predictor_{session_id}.pt")


def load_model(session_id=None):
    model = get_model_instance_segmentation()
    if session_id is not None:
        model.roi_heads.box_predictor.load_state_dict(torch.load(
            f"./checkpoints/rcnn_box_predictor_{session_id}.pt",
            map_location=None if torch.cuda.is_available() else torch.device('cpu')
        ))
    return model
