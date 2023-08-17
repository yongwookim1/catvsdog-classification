import torch
import torch.nn.functional as F

from data_loader.data_loaders import data_loader
from models.ResNet50 import ResNet50


def inference(model, test_loader):
    model.eval()
    for batch_in, img_name in test_loader:
        batch_in = batch_in.to(device)

        y_pred = model(batch_in)
        y_pred = F.softmax(y_pred, dim=1)
        _, preds = torch.max(y_pred, 1)

        print(
            f'"{img_name[0]}" is ' + "a cat"
            if preds[0] == 1
            else f'"{img_name[0]}" is ' + "a dog"
        )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("/opt/ml/checkpoints/catvsdog.pt")
    model.to(device)
    inference(model, data_loader("test"))
