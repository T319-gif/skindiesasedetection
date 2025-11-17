import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# --- Load the trained H5 model ---
model = tf.keras.models.load_model("skin_cancer_model_rgb_v2.h5")

# --- Full descriptive class mapping ---
class_indices = {
    0: "Actinic keratoses and intraepithelial carcinoma / Bowen's disease (akiec)",
    1: "Basal cell carcinoma (bcc)",
    2: "Benign keratosis-like lesions (bkl)",
    3: "Dermatofibroma (df)",
    4: "Melanoma (mel)",
    5: "Melanocytic nevi (nv)",
    6: "Vascular lesions (vasc)"
}

def predict_image(img_path, top_k=3):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)[0]

    # Get top K predictions
    top_indices = preds.argsort()[-top_k:][::-1]
    print(f"Top {top_k} predictions for {img_path}:")

    for i in top_indices:
        class_name = class_indices[i]
        probability = preds[i] * 100
        print(f"  {class_name}: {probability:.2f}%")

# --- Example usage ---
predict_image(r"C:\Users\TZZS\Downloads\archive\HAM10000_images_part_2\ISIC_0034264.jpg")
