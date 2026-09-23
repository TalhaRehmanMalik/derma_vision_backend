
# """
# utils/inference.py
# MobileNetV2 model loading and image inference.
# Includes skin classifier for OOD (non-skin image) rejection.

# Two-stage pipeline:
#     1. Skin classifier — reject non-skin images immediately
#     2. Cancer classifier — classify into 4 skin cancer types
# """
# import json
# import os
# import uuid

# import numpy as np
# from PIL import Image

# from utils.logger import get_logger

# logger = get_logger("inference")

# # ── Singletons ────────────────────────────────────────────────
# _cancer_model  = None
# _skin_model    = None
# _class_names   = None

# CONFIDENCE_THRESHOLD = 0.60
# SKIN_THRESHOLD       = 0.50   # below this = non-skin image
# ALLOWED_EXTENSIONS   = {"jpg", "jpeg", "png"}


# def load_model(model_path: str, mapping_path: str) -> None:
#     """
#     Load cancer classifier weights at startup.
#     Architecture rebuilt in code — version independent.
#     """
#     global _cancer_model, _class_names

#     if not os.path.exists(model_path):
#         logger.warning(f"Cancer model not found at '{model_path}'")
#         return

#     import tensorflow as tf
#     from tensorflow.keras import layers, Model
#     from tensorflow.keras.applications import MobileNetV2

#     base    = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
#     inputs  = tf.keras.Input(shape=(224, 224, 3))
#     x       = base(inputs, training=False)
#     x       = layers.GlobalAveragePooling2D()(x)
#     x       = layers.BatchNormalization()(x)
#     x       = layers.Dense(256, activation="relu")(x)
#     x       = layers.Dropout(0.4)(x)
#     x       = layers.Dense(128, activation="relu")(x)
#     x       = layers.Dropout(0.3)(x)
#     outputs = layers.Dense(4, activation="softmax")(x)
#     _cancer_model = Model(inputs, outputs)
#     _cancer_model.load_weights(model_path)

#     with open(mapping_path, "r") as f:
#         mapping = json.load(f)
#     _class_names = mapping["class_names"]

#     logger.info(f"Cancer model loaded. Classes: {_class_names}")


# def load_skin_classifier(skin_model_path: str) -> None:
#     """
#     Load binary skin classifier weights at startup.
#     Output: sigmoid — > 0.5 = skin, < 0.5 = non-skin
#     """
#     global _skin_model

#     if not os.path.exists(skin_model_path):
#         logger.warning(f"Skin classifier not found at '{skin_model_path}' — OOD check disabled")
#         return

#     import tensorflow as tf
#     from tensorflow.keras import layers, Model
#     from tensorflow.keras.applications import MobileNetV2

#     base_s    = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
#     inputs_s  = tf.keras.Input(shape=(224, 224, 3))
#     x_s       = base_s(inputs_s, training=False)
#     x_s       = layers.GlobalAveragePooling2D()(x_s)
#     x_s       = layers.Dense(128, activation="relu")(x_s)
#     x_s       = layers.Dropout(0.3)(x_s)
#     outputs_s = layers.Dense(1, activation="sigmoid")(x_s)
#     _skin_model = Model(inputs_s, outputs_s)
#     _skin_model.load_weights(skin_model_path)

#     logger.info("Skin classifier loaded.")


# def allowed_file(filename: str) -> bool:
#     return (
#         "." in filename
#         and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
#     )


# def save_image(file_bytes: bytes, original_filename: str, upload_folder: str) -> str:
#     """Save image with UUID filename. Returns saved filename."""
#     ext      = original_filename.rsplit(".", 1)[1].lower()
#     filename = f"{uuid.uuid4().hex}.{ext}"
#     path     = os.path.join(upload_folder, filename)
#     with open(path, "wb") as f:
#         f.write(file_bytes)
#     logger.info(f"Image saved: {filename}")
#     return filename


# def _preprocess(image_path: str) -> np.ndarray:
#     """Resize to 224x224, normalize to 0-1, add batch dim."""
#     img = Image.open(image_path).convert("RGB")
#     img = img.resize((224, 224))
#     arr = np.array(img, dtype=np.float32) / 255.0
#     return np.expand_dims(arr, axis=0)


# def predict(image_path: str) -> dict:
#     """
#     Two-stage inference:
#         Stage 1: Skin classifier — reject non-skin images
#         Stage 2: Cancer classifier — 4-class prediction

#     Returns:
#         predicted_class, confidence, all_probabilities,
#         inconclusive, non_skin (bool)
#     """
#     if _cancer_model is None:
#         raise RuntimeError(
#             "Cancer model not loaded. Place model_weights.weights.h5 in ml_models/ and restart."
#         )

#     tensor = _preprocess(image_path)

#     # ── Stage 1: Skin check ───────────────────────────────────
#     if _skin_model is not None:
#         skin_prob = float(_skin_model.predict(tensor, verbose=0)[0][0])
#         logger.info(f"Skin probability: {skin_prob:.4f}")

#         if skin_prob < SKIN_THRESHOLD:
#             logger.warning(f"Non-skin image rejected — skin_prob={skin_prob:.4f}")
#             return {
#                 "predicted_class":   "Invalid Input",
#                 "confidence":        round(skin_prob, 4),
#                 "all_probabilities": {},
#                 "inconclusive":      True,
#                 "non_skin":          True,
#                 "message":           "Non-skin image detected. Please upload a dermoscopic skin image.",
#             }

#     # ── Stage 2: Cancer classification ───────────────────────
#     probs = _cancer_model.predict(tensor, verbose=0)[0]
#     idx   = int(np.argmax(probs))
#     conf  = float(probs[idx])

#     # Entropy check
#     entropy       = -np.sum(probs * np.log(probs + 1e-8))
#     max_entropy   = np.log(4)
#     entropy_ratio = entropy / max_entropy

#     all_probs = {
#         _class_names[i]: round(float(probs[i]), 4)
#         for i in range(len(_class_names))
#     }

#     # Hard cutoff
#     if conf < 0.40:
#         return {
#             "predicted_class":   "Unknown",
#             "confidence":        round(conf, 4),
#             "all_probabilities": all_probs,
#             "inconclusive":      True,
#             "non_skin":          False,
#             "message":           "Result inconclusive — please consult a dermatologist.",
#         }

#     # Uncertainty cutoff
#     if conf < 0.50 and entropy_ratio > 0.65:
#         return {
#             "predicted_class":   "Unknown",
#             "confidence":        round(conf, 4),
#             "all_probabilities": all_probs,
#             "inconclusive":      True,
#             "non_skin":          False,
#             "message":           "Result inconclusive — please consult a dermatologist.",
#         }

#     inconclusive = conf < CONFIDENCE_THRESHOLD

#     logger.info(f"Prediction: {_class_names[idx]} | confidence: {conf:.2%} | entropy: {entropy_ratio:.3f}")

#     return {
#         "predicted_class":   _class_names[idx],
#         "confidence":        round(conf, 4),
#         "all_probabilities": all_probs,
#         "inconclusive":      inconclusive,
#         "non_skin":          False,
#         "message":           (
#             "Result inconclusive — please consult a dermatologist."
#             if inconclusive else
#             "AI screening tool only. Does NOT replace professional medical diagnosis."
#         ),
#     }





"""
utils/inference.py
MobileNetV2 model loading and image inference.
Includes skin classifier for OOD (non-skin image) rejection.

Two-stage pipeline:
    1. Skin classifier — reject non-skin images immediately
    2. Cancer classifier — classify into 4 skin cancer types
"""
import json
import os
import uuid

import numpy as np
from PIL import Image

from utils.logger import get_logger

logger = get_logger("inference")

# ── Singletons ────────────────────────────────────────────────
_cancer_model  = None
_skin_model    = None
_class_names   = None

CONFIDENCE_THRESHOLD = 0.60
SKIN_THRESHOLD       = 0.50   # below this = non-skin image
ALLOWED_EXTENSIONS   = {"jpg", "jpeg", "png"}


def load_model(model_path: str, mapping_path: str) -> None:
    """
    Load cancer classifier weights at startup.
    Architecture rebuilt in code — version independent.
    """
    global _cancer_model, _class_names

    if not os.path.exists(model_path):
        logger.warning(f"Cancer model not found at '{model_path}'")
        return

    import tensorflow as tf
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import MobileNetV2

    try:
        base    = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
        inputs  = tf.keras.Input(shape=(224, 224, 3))
        x       = base(inputs, training=False)
        x       = layers.GlobalAveragePooling2D()(x)
        x       = layers.BatchNormalization()(x)
        x       = layers.Dense(256, activation="relu")(x)
        x       = layers.Dropout(0.4)(x)
        x       = layers.Dense(128, activation="relu")(x)
        x       = layers.Dropout(0.3)(x)
        outputs = layers.Dense(4, activation="softmax")(x)
        _cancer_model = Model(inputs, outputs)
        _cancer_model.load_weights(model_path)

        with open(mapping_path, "r") as f:
            mapping = json.load(f)
        _class_names = mapping["class_names"]

        logger.info(f"Cancer model loaded successfully. Classes: {_class_names}")
    except Exception as e:
        logger.error(f"Failed to load cancer model weights: {e}")


def load_skin_classifier(skin_model_path: str) -> None:
    """
    Load binary skin classifier weights at startup.
    Output: sigmoid — > 0.5 = skin, < 0.5 = non-skin
    """
    global _skin_model

    if not os.path.exists(skin_model_path):
        logger.warning(f"Skin classifier not found at '{skin_model_path}' — OOD check disabled")
        return

    import tensorflow as tf
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import MobileNetV2

    try:
        base_s    = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
        inputs_s  = tf.keras.Input(shape=(224, 224, 3))
        x_s       = base_s(inputs_s, training=False)
        x_s       = layers.GlobalAveragePooling2D()(x_s)
        x_s       = layers.BatchNormalization()(x_s)
        x_s       = layers.Dense(256, activation="relu")(x_s)
        x_s       = layers.Dropout(0.4)(x_s)
        x_s       = layers.Dense(128, activation="relu")(x_s)
        x_s       = layers.Dropout(0.3)(x_s)
        outputs_s = layers.Dense(1, activation="sigmoid")(x_s)
        _skin_model = Model(inputs_s, outputs_s)
        _skin_model.load_weights(skin_model_path)

        logger.info("Skin classifier loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load skin classifier weights: {e}")


def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def save_image(file_bytes: bytes, original_filename: str, upload_folder: str) -> str:
    """Save image with UUID filename. Returns saved filename."""
    ext      = original_filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    path     = os.path.join(upload_folder, filename)
    with open(path, "wb") as f:
        f.write(file_bytes)
    logger.info(f"Image saved: {filename}")
    return filename


def _preprocess(image_path: str) -> np.ndarray:
    """Resize to 224x224, normalize to 0-1, add batch dim."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict(image_path: str) -> dict:
    """
    Two-stage inference:
        Stage 1: Skin classifier — reject non-skin images
        Stage 2: Cancer classifier — 4-class prediction

    Returns:
        predicted_class, confidence, all_probabilities,
        inconclusive, non_skin (bool)
    """
    if _cancer_model is None:
        raise RuntimeError(
            "Cancer model not loaded. Place model file in ml_models/ and restart."
        )

    tensor = _preprocess(image_path)

    # ── Stage 1: Skin check ───────────────────────────────────
    if _skin_model is not None:
        skin_prob = float(_skin_model.predict(tensor, verbose=0)[0][0])
        logger.info(f"Skin probability: {skin_prob:.4f}")

        if skin_prob < SKIN_THRESHOLD:
            logger.warning(f"Non-skin image rejected — skin_prob={skin_prob:.4f}")
            return {
                "predicted_class":   "Invalid Input",
                "confidence":        round(skin_prob, 4),
                "all_probabilities": {},
                "inconclusive":      True,
                "non_skin":          True,
                "message":           "Non-skin image detected. Please upload a dermoscopic skin image.",
            }
    else:
        logger.warning("Skin classifier is not loaded! OOD rejection bypassed.")

    # ── Stage 2: Cancer classification ───────────────────────
    probs = _cancer_model.predict(tensor, verbose=0)[0]
    idx   = int(np.argmax(probs))
    conf  = float(probs[idx])

    # Entropy check
    entropy       = -np.sum(probs * np.log(probs + 1e-8))
    max_entropy   = np.log(4)
    entropy_ratio = entropy / max_entropy

    all_probs = {
        _class_names[i]: round(float(probs[i]), 4)
        for i in range(len(_class_names))
    }

    # Hard cutoff
    if conf < 0.40:
        return {
            "predicted_class":   "Unknown",
            "confidence":        round(conf, 4),
            "all_probabilities": all_probs,
            "inconclusive":      True,
            "non_skin":          False,
            "message":           "Result inconclusive — please consult a dermatologist.",
        }

    # Uncertainty cutoff
    if conf < 0.50 and entropy_ratio > 0.65:
        return {
            "predicted_class":   "Unknown",
            "confidence":        round(conf, 4),
            "all_probabilities": all_probs,
            "inconclusive":      True,
            "non_skin":          False,
            "message":           "Result inconclusive — please consult a dermatologist.",
        }

    inconclusive = conf < CONFIDENCE_THRESHOLD

    logger.info(f"Prediction: {_class_names[idx]} | confidence: {conf:.2%} | entropy: {entropy_ratio:.3f}")

    return {
        "predicted_class":   _class_names[idx],
        "confidence":        round(conf, 4),
        "all_probabilities": all_probs,
        "inconclusive":      inconclusive,
        "non_skin":          False,
        "message":           (
            "Result inconclusive — please consult a dermatologist."
            if inconclusive else
            "AI screening tool only. Does NOT replace professional medical diagnosis."
        ),
    }