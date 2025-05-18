from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
from io import BytesIO
from load_model import read_image

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'Mohon masukkan gambar yang valid.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'Tidak ada file gambar yang dimasukkan.'}), 400

    try:
        # Membaca gambar dan mengonversinya menjadi array NumPy
        image = Image.open(file.stream)
        image_np = np.array(image)

        # Memproses gambar dengan fungsi read_image
        result = read_image(image_np)

        return jsonify(result), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)