import torch
import torch.nn.functional as F

import albumentations as A
from albumentations.pytorch import ToTensorV2

import streamlit as st

import io

import numpy as np
from PIL import Image

from models.ResNet50 import ResNet50

st.set_page_config(layout="wide")


def transform_image(image_bytes: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")
    image_array = np.array(image)

    test_transforms = A.Compose(
        [
            A.Resize(height=256, width=256),
            A.CLAHE(
                always_apply=False, p=1.0, clip_limit=(4, 4), tile_grid_size=(8, 8)
            ),
            A.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
                max_pixel_value=255.0,
                always_apply=True,
            ),
            ToTensorV2(always_apply=True),
        ]
    )
    return test_transforms(image=image_array)["image"].unsqueeze(0)


def get_prediction(model, image_bytes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformed_image = transform_image(image_bytes=image_bytes).to(device)
    outputs = model.forward(transformed_image)
    outputs = F.softmax(outputs, dim=1)
    _, y_hat = torch.max(outputs, 1)
    return transformed_image, y_hat


def main():
    st.title("Cat and Dog Classification Model")

    model = torch.load("/opt/ml/checkpoints/catvsdog.pt")
    model.eval()

    uploaded_file = st.file_uploader(
        "Upload cat or dog image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))

        st.image(image, caption="Uploaded Image")
        _, y_hat = get_prediction(model, image_bytes)
        label = y_hat[0]

        if label == 1:
            st.header("It is a cat")
        else:
            st.header("It is a dog")


if __name__ == "__main__":
    main()
