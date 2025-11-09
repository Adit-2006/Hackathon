from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models

def ClassifyImages(image_path) -> str:
    """
    This Function takes a file path as a input and gives out the predicted label of the given file;
    """
    # 1. Load a pretrained image classification model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.eval()   # Set model to evaluation mode

    # 2. Preprocessing pipeline: resize → convert to tensor → normalize
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 3. Load your image
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0)  # Add batch dimension

    # 4. Classify image
    with torch.no_grad():
        output = model(img_t)

    # 5. Load label names
    labels_url = models.ResNet50_Weights.DEFAULT.meta["categories"]

    # 6. Get the highest scoring result
    _, predicted_index = output.max(1)
    predicted_label = labels_url[predicted_index]
    return predicted_label

image_path = "dog.avif"

print("👉 The image likely contains:", ClassifyImages(image_path))
