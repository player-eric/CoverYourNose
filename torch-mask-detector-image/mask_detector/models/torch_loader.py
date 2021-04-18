import torch


def load_torch_model(torch_model_path):
    """
    Load the model.
    :param torch_model_path: model to torch model.
    :return: model
    """
    model = torch.load(torch_model_path)
    return model


def torch_inference(model, img_arr):
    """
    Receive an image array and run inference
    :param model: torch model.
    :param img_arr: 3D numpy array, RGB order.
    :return:
    """
    if torch.cuda.is_available():
        dev = 'cuda:0'
    else:
        dev = 'cpu'
    device = torch.device(dev)
    model.to(device)
    input_tensor = torch.tensor(img_arr).float().to(device)
    y_bboxes, y_scores, = model.forward(input_tensor)
    return y_bboxes.detach().cpu().numpy(), y_scores.detach().cpu().numpy()
