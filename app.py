from flask import Flask, request, jsonify, send_from_directory, url_for
from PIL import Image
import numpy as np
from io import BytesIO
from load_model import read_image
import os
from dotenv import load_dotenv
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import uuid
import json 

load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = 'images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class ScanLog(db.Model):
    __tablename__ = 'scan'
    __table_args__ = {'schema': 'log'}

    id = db.Column(db.Integer, primary_key=True)
    image = db.Column(db.Text, nullable=False) 
    result = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.now)

    def __repr__(self):
        return f"<ScanLog {self.id}>"

with app.app_context():
    db.create_all()

api_key = os.environ.get('OCR_SPACE_API_KEY')

if api_key:
    print(f"API Key dari .env: {api_key}")
else:
    print("Error: OCR_SPACE_API_KEY not found")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'Please insert valid picture.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'There are no valid image that you inserted'}), 400

    try:
        image_bytes = file.read()
        image = Image.open(BytesIO(image_bytes))
        image_np = np.array(image)

        filename = f"{uuid.uuid4()}.{file.filename.rsplit('.', 1)[1].lower()}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        image.save(file_path)

        result = read_image(image_np, api_key)

        result_to_save = json.dumps(result) 
        new_log = ScanLog(image=filename, result=result_to_save)
        db.session.add(new_log)
        db.session.commit()

        # image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename).replace(os.sep, '/')

        response_data = result
        # response_data['image'] = image_path

        return jsonify(response_data), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/log/<int:log_id>', methods=['GET'])
def get_log_details(log_id):
    scan_log = ScanLog.query.get(log_id)
    if not scan_log:
        return jsonify({'status': 'error', 'message': 'Log ID tidak ditemukan.'}), 404

    try:
        filename = scan_log.image
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename).replace(os.sep, '/')

        parsed_result = json.loads(scan_log.result)

        timestamp_str = scan_log.timestamp.isoformat() if scan_log.timestamp else None

        return jsonify({
            'id': scan_log.id,
            'image_url': image_path,
            'result': parsed_result,
            'timestamp': timestamp_str
        }), 200

    except json.JSONDecodeError as e:
        return jsonify({'status': 'error', 'message': f'Failed to parsed OCR result: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Failed to get log details: {str(e)}'}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)