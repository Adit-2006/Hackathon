import os
import uuid
from Classify_image import ClassifyImages

CATEGORY_MAP = {
    "cat": "animals",
    "dog": "animals",
    "lion": "animals",
    "car": "vehicles",
    "bus": "vehicles"
}

BASE_DIR = "storage"

def save_classified_file(image_path: str, predicted_label: str):
    """
    Creates category folder based on predicted label
    and saves the file with UUID.
    """

    # 1) File base + extension
    original_name = os.path.basename(image_path)
    file_name, file_ext = os.path.splitext(original_name)
    file_ext = file_ext.replace(".", "")

    # 2) Determine category
    category = CATEGORY_MAP.get(predicted_label.lower(), predicted_label.lower())

    # 3) Create folder
    folder_path = os.path.join(BASE_DIR, category)
    os.makedirs(folder_path, exist_ok=True)

    # 4) Unique name
    new_name = f"{file_name}_{uuid.uuid4().hex}.{file_ext}"
    dest_path = os.path.join(folder_path, new_name)

    # 5) Save file (copy/move)
    with open(image_path, "rb") as src:
        data = src.read()

    with open(dest_path, "wb") as dst:
        dst.write(data)

    return dest_path

image_path = "dog.avif"
predicted = ClassifyImages(image_path)

saved_path = save_classified_file(image_path, predicted)
print("✅ File stored at:", saved_path)