import cv2
import numpy as np

#tambah gaussian filter
def preprocess(image):
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image_resized = cv2.resize(image_np, (640, 640))
    return image_resized

def preprocess_for_ocr(image_np):
    # Pastikan gambar dalam grayscale
    if len(image_np.shape) == 3:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    # Equalize lighting
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image_np)

    # Blur untuk meredam noise
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    return blurred