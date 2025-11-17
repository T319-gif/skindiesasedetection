
import os
import base64
import json
import traceback
import hashlib
from io import BytesIO
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Use environment variable for model path if available, otherwise default.
MODEL_PATH = os.environ.get('MODEL_PATH', 'skin_condition_model.h5')
# Optional: path to class labels JSON; if not found, a sensible default will be used.
CLASS_LABELS_PATH = os.environ.get('CLASS_LABELS_PATH', 'class_labels.json')
# Define the image size your model expects
IMG_WIDTH = 224
IMG_HEIGHT = 224
# Preprocess switch: 'efficientnet' (default) or 'rescale'
# Your model uses EfficientNetV2B0, so this default matches training.
PREPROCESS = os.environ.get('PREPROCESS', 'efficientnet').lower()

# --- Flask App Initialization ---
app = Flask(__name__)
# Enable Cross-Origin Resource Sharing to allow requests from the website
CORS(app) 

# --- Load the Model ---
# Load your trained model.
# This is done once when the server starts.
try:
    print(f"Loading model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# --- Class Labels ---
# Try to load class labels from a JSON file; fallback to a default.
# For HAM10000, the usual alphabetical order is: akiec, bcc, bkl, df, mel, nv, vasc
_DEFAULT_CLASS_LABELS = [
    'akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'
]

def load_class_labels():
    try:
        if CLASS_LABELS_PATH and os.path.exists(CLASS_LABELS_PATH):
            with open(CLASS_LABELS_PATH, 'r') as f:
                labels = json.load(f)
            if isinstance(labels, list) and all(isinstance(x, str) for x in labels):
                print(f"✅ Loaded {len(labels)} class labels from {CLASS_LABELS_PATH}")
                return labels
            else:
                print("⚠️ class_labels.json is not a list of strings; using default labels.")
        else:
            print("ℹ️ No class_labels.json found; using default labels.")
    except Exception as e:
        print(f"⚠️ Failed to load class labels: {e}. Using default labels.")
    return _DEFAULT_CLASS_LABELS

CLASS_LABELS = load_class_labels()

# --- Image Preprocessing ---
def preprocess_image(image_data: str):
    """
    Preprocesses the base64 encoded image data to the format your model expects.
    
    IMPORTANT: You MUST adjust this function to match the preprocessing steps
    used in your training notebook (e.g., resizing, normalization).
    """
    # Decode the base64 string
    base64_string = image_data.split(",")[1]
    img_bytes = base64.b64decode(base64_string)

    # Log a short checksum to confirm unique images are received
    md5 = hashlib.md5(img_bytes).hexdigest()
    print("Image MD5:", md5)
    
    # Open the image
    img = Image.open(BytesIO(img_bytes))
    
    # Ensure image is RGB
    img = img.convert('RGB')
    
    # Resize the image to what the model expects
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    
    # Convert image to a numpy array
    img_array = np.array(img).astype('float32')
    
    # --- IMPORTANT: NORMALIZE YOUR IMAGE ---
    if PREPROCESS.startswith('eff'):
        try:
            from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as eff_preprocess
            img_array = eff_preprocess(img_array)
            print("Using EfficientNetV2 preprocess_input")
        except Exception:
            img_array = img_array / 255.0
            print("EfficientNetV2 preprocess unavailable; falling back to rescale/255")
    else:
        img_array = img_array / 255.0
    
    # Add a batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# --- API Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model is not loaded.'}), 500
        
    if not request.json or 'photoDataUri' not in request.json:
        return jsonify({'error': 'No image data provided.'}), 400

    try:
        # Get image data from the request
        image_data = request.json['photoDataUri']
        print("Received image payload; length:", len(image_data))
        
        # Preprocess the image
        try:
            processed_image = preprocess_image(image_data)
            print("Preprocess OK; shape:", getattr(processed_image, 'shape', None))
        except Exception as pe:
            print("❌ Preprocess error:")
            traceback.print_exc()
            return jsonify({'error': 'Preprocess failed'}), 500
        
        # Make a prediction
        try:
            prediction_result = model.predict(processed_image)
            print("Predict OK; raw shape:", getattr(prediction_result, 'shape', None))
            try:
                import numpy as _np
                print("Pred vector (first 7):", [_np.round(x, 4).item() for x in prediction_result[0][:7]])
                top_i = int(_np.argmax(prediction_result[0]))
                print("Top index:", top_i, "label:", CLASS_LABELS[top_i] if top_i < len(CLASS_LABELS) else top_i)
            except Exception:
                pass
        except Exception as me:
            print("❌ Model predict error:")
            traceback.print_exc()
            return jsonify({'error': 'Model predict failed'}), 500
        
        # --- IMPORTANT: FORMAT THE PREDICTION ---
        predictions_formatted = []
        try:
            for i, prob in enumerate(prediction_result[0]):
                if i < len(CLASS_LABELS):
                    predictions_formatted.append({
                        "condition": CLASS_LABELS[i],
                        "probability": float(prob)
                    })
        except Exception:
            print("❌ Formatting predictions error:")
            traceback.print_exc()
            return jsonify({'error': 'Format predictions failed'}), 500

        return jsonify({"predictions": predictions_formatted})

    except Exception:
        print("❌ Error during prediction root:")
        traceback.print_exc()
        return jsonify({'error': 'Failed to process image.'}), 500

# --- Run the App ---
if __name__ == '__main__':
    # Runs the Flask server on localhost with port 5001.
    # The port must match the one in your .env.local file.
    app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)

