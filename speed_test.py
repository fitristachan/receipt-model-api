import requests
import time
from PIL import Image
from io import BytesIO
import os

# ---------- CONFIG ----------
url = 'http://127.0.0.1:5000/predict'
input_image_path = 'contoh-1.jpeg'
log_dir = 'Log'
os.makedirs(log_dir, exist_ok=True)
# ----------------------------

def compress_image_file(input_path):
    img = Image.open(input_path)
    log_dir = 'Log'
    os.makedirs(log_dir, exist_ok=True)

    def compress_loop(image):
        quality = 100
        while True:
            stream = BytesIO()
            image.save(stream, format='JPEG', quality=quality)
            byte_array = stream.getvalue()
            if len(byte_array) <= 1000 * 1024 or quality <= 80:
                return byte_array
            quality -= 5

    byte_array = compress_loop(img)

    if len(byte_array) > 1000 * 1024:
        width, height = img.size
        resized_img = img.resize((width * 0.8, height * 0.8), Image.ANTIALIAS)
        byte_array = compress_loop(resized_img)

    compressed_path = os.path.join(log_dir, f"compressed_{os.path.basename(input_path)}")
    with open(compressed_path, "wb") as f:
        f.write(byte_array)

    return compressed_path

compressed_image_path = compress_image_file(input_image_path)


with open(compressed_image_path, 'rb') as f:
    files = {'image': f}
    
    total_start = time.time()

    send_start = time.time()
    response = requests.post(url, files=files)
    send_end = time.time()
    
    total_end = time.time()

try:
    result = response.json()
except Exception as e:
    result = {"error": f"Invalid response: {e}"}

print(f"Total waktu    : {total_end - total_start:.2f} detik")
print(f"Waktu kirim    : {send_end - send_start:.2f} detik")
print(f"Ukuran gambar  : {os.path.getsize(compressed_image_path) / 1024:.2f} KB")
print("Response API   :", result)
