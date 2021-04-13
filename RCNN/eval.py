from util import model_to_device
from model import load_model

if __name__ == "__main__":
    model = load_model()
    model.eval()
    model_to_device(model)
