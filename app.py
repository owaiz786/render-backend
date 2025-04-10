from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from estimator import ImprovedGlucoseEstimator
import os

app = Flask(__name__)
CORS(app)

# Load the estimator once
estimator = ImprovedGlucoseEstimator()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({"error": "No image provided"}), 400

    try:
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        img_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"error": f"Invalid image data: {str(e)}"}), 400

    processed_frame, glucose, eye_frame = estimator.process_frame(frame)

    if glucose is None:
        return jsonify({"message": "Collecting data..."}), 202
    else:
        return jsonify({"glucose": float(glucose)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))