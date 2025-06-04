from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
from io import BytesIO
from load_model import read_image 
import os 
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

api_key = os.environ.get('OCR_SPACE_API_KEY')

if api_key:
    print(f"API Key dari .env: {api_key}")
else:
    print("Error: OCR_SPACE_API_KEY tidak ditemukan di .env atau environment.")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'Mohon masukkan gambar yang valid.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'Tidak ada file gambar yang dimasukkan.'}), 400

    try:
        # Membaca gambar
        image_bytes = file.read() # Baca sebagai bytes untuk dikirim ke API atau diproses
        image = Image.open(BytesIO(image_bytes))
        image_np = np.array(image)

        # Memproses gambar dengan fungsi read_image yang sudah dimodifikasi
        # Kita akan meneruskan API key ke fungsi read_image
        result = read_image(image_np, api_key) # Modifikasi di sini

        return jsonify(result), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)