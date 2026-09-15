"""
utils/inference.py
Loads the MobileNetV2 model once at startup and reuses it for every prediction.
TensorFlow is imported lazily (inside load_model) to keep startup fast if model is missing.
"""
import os
import json
import uuid
import numpy as np
from PIL import Image

from utils.logger import get_logger

logger = get_logger("inference")

# Module-level singletons — populated by load_model()
_model       = None
_mapping     = None
_class_names = None

CONFIDENCE_THRESHOLD = 0.45
ENTROPY_THRESHOLD    = 0.70
ALLOWED_EXTENSIONS   = {"jpg", "jpeg", "png"}


# def load_model(model_path: str, mapping_path: str):
#     """
#     Called once in main.py at startup.
#     Loads the Keras .h5 model and the class-name mapping JSON.
#     If the model file is missing the server still starts — predict() will raise RuntimeError.
#     """
#     global _model, _mapping, _class_names

#     if not os.path.exists(model_path):
#         # logger.warning(f"Model file not found at '{model_path}' — place .h5 in ml_models/")
#         logger.warning(f"Model file not found at '{model_path}' — place .keras file in ml_models/")
#         return

#     # TF import is slow — kept here so it does not delay startup when model is absent
#     import tensorflow as tf
#     _model = tf.keras.models.load_model(model_path)

#     with open(mapping_path, "r") as f:
#         _mapping = json.load(f)

#     _class_names = _mapping["class_names"]
#     logger.info(f"Model loaded. Classes: {_class_names}")

def load_model(model_path: str, mapping_path: str):
    global _model, _mapping, _class_names

    if not os.path.exists(model_path):
        logger.warning(f"Weights file not found at '{model_path}'")
        return

    import tensorflow as tf
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import MobileNetV2

    # Build fresh architecture — no config loading, no version issues
    base    = MobileNetV2(input_shape=(224,224,3), include_top=False, weights=None)
    inputs  = tf.keras.Input(shape=(224,224,3))
    x       = base(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dense(256, activation="relu")(x)
    x       = layers.Dropout(0.4)(x)
    x       = layers.Dense(128, activation="relu")(x)
    x       = layers.Dropout(0.3)(x)
    outputs = layers.Dense(4, activation="softmax")(x)
    _model  = Model(inputs, outputs)

    # Load only weights — no config, no version conflict
    _model.load_weights(model_path)

    with open(mapping_path, "r") as f:
        _mapping = json.load(f)

    _class_names = _mapping["class_names"]
    logger.info(f"Model loaded via weights. Classes: {_class_names}")



def allowed_file(filename: str) -> bool:
    """Return True if the file extension is in the allowed set."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_image(file_bytes: bytes, original_filename: str, upload_folder: str) -> str:
    """
    Save raw bytes to disk under a UUID-based filename.
    UUID prevents filename collisions and path-traversal attacks.
    Returns the new filename (not the full path).
    """
    ext      = original_filename.rsplit(".", 1)[1].lower()
    new_name = f"{uuid.uuid4().hex}.{ext}"
    path     = os.path.join(upload_folder, new_name)
    with open(path, "wb") as f:
        f.write(file_bytes)
    logger.info(f"Image saved as {new_name}")
    return new_name


def preprocess(image_path: str) -> np.ndarray:
    """
    Resize to 224x224 and normalize pixel values to [0, 1].
    Returns a batch tensor of shape (1, 224, 224, 3) — MobileNetV2 input format.
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict(image_path: str) -> dict:
    """
    Run inference on a single image file.
    Returns predicted class, confidence, all class probabilities, and inconclusive flag.
    Raises RuntimeError if model was not loaded.
    """
    if _model is None:
        raise RuntimeError(
            # "Model not loaded. Place derma_vision_mobilenetv2.h5 in ml_models/ and restart."
            # "Model not loaded. Place derma_vision_model.keras in ml_models/ and restart."

            "Model not loaded. Place derma_vision_model_fixed.keras in ml_models/ and restart."
        )

    tensor = preprocess(image_path)
    probs = np.asarray(_model.predict(tensor, verbose=0)[0], dtype=np.float32)
    entropy = -np.sum(probs * np.log(probs + 1e-8))
    max_entropy = np.log(4)
    entropy_ratio = entropy / max_entropy
    max_conf = float(np.max(probs))
    predicted_class_idx = int(np.argmax(probs))

    if entropy_ratio > ENTROPY_THRESHOLD and max_conf < CONFIDENCE_THRESHOLD:
        return {
            "status": "rejected",
            "predicted_class": "Unknown / Out of Scope",
            "confidence": round(max_conf, 4),
            "entropy_ratio": round(float(entropy_ratio), 4),
            "inconclusive": True,
            "ood_detected": True,
            "message": "The provided image does not match the trained 4 skin lesion classes."
        }

    all_probs = {
        str(_class_names[i]): round(float(probs[i]), 4)
        for i in range(len(_class_names))
    }

    logger.info(f"Prediction: {_class_names[predicted_class_idx]} | confidence: {max_conf:.2%}")

    return {
        "predicted_class":   str(_class_names[predicted_class_idx]),
        "confidence":        round(max_conf, 4),
        "all_probabilities": all_probs,
        "inconclusive":      False,
    }