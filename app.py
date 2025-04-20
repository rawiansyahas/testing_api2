import os
import glob
from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
import gdown

app = Flask(__name__)

DRIVE_ID = "1HmnlXmarGgCyve7b3NkYGRFb3DXcoPIU"
MODEL_PATH = "ms1mv2_r50.onnx"

if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={DRIVE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])

input_name = session.get_inputs()[0].name

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=False)


def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Compute softmax probabilities over last dimension.
    """
    exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


def preprocess(image: Image.Image) -> np.ndarray:
    """
    Preprocess a cropped face image:
      - Resize to 112x112
      - HWC -> CHW
      - Mean subtraction [120,105,90]
      - Batch dimension
    """
    img = image.resize((112, 112))
    arr = np.array(img, dtype=np.float32).transpose(2, 0, 1)
    arr -= np.array([120, 105, 90], dtype=np.float32).reshape(3,1,1)
    return np.expand_dims(arr, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    # Validate upload
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    try:
        image = Image.open(request.files['file'].stream).convert('RGB')
    except Exception as e:
        return jsonify({'error': 'Invalid image file', 'details': str(e)}), 400

    # Detect face
    boxes, _ = mtcnn.detect(image)
    if boxes is None or len(boxes) == 0:
        return jsonify({'error': 'No face detected'}), 400
    x1, y1, x2, y2 = boxes[0]
    face = image.crop((int(x1), int(y1), int(x2), int(y2)))

    # Run ONNX model to get logits
    input_tensor = preprocess(face)
    logits = session.run(None, {input_name: input_tensor})[0]  # shape (1, num_classes)

    # Compute probabilities and prediction via numpy
    probs = softmax(logits)
    class_idx = int(np.argmax(probs, axis=1)[0])
    confidence = float(probs[0, class_idx])

    return jsonify({
        'predicted_class': class_idx,
        'confidence': confidence
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)