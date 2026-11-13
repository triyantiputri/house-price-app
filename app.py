"""Aplikasi Prediksi Harga Rumah - Tugas Besar 3 Machine Learning.
Flask memuat model hasil training dari Google Colab dan menyediakan
form web + endpoint API /predict."""
import json
import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "model", "model.pth"))
FEATURES_PATH = os.path.join(BASE_DIR, "model", "feature_columns.json")
# Biarkan False. Notebook memakai TransformedTargetRegressor yang sudah mengembalikan
# prediksi ke harga asli. True hanya jika model dilatih manual dengan np.log1p(price).
LOG_TARGET = os.environ.get("LOG_TARGET", "false").lower() == "true"

API_KEY = os.environ.get("API_KEY", "")  # opsional; samakan dengan Script Property di Apps Script

app = Flask(__name__)


def load_model(path):
    """Coba joblib (sklearn/xgboost), lalu pickle, lalu torch."""
    if not os.path.exists(path):
        return None
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        pass
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        pass
    import torch  # hanya jika model berupa PyTorch
    return torch.load(path, map_location="cpu", weights_only=False)


model = load_model(MODEL_PATH)
with open(FEATURES_PATH, encoding="utf-8") as f:
    FEATURE_COLUMNS = json.load(f)

YES_NO = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
NUMERIC = ["area", "bedrooms", "bathrooms", "stories", "parking"]


FURNISHING = ("furnished", "semi-furnished", "unfurnished")


def build_features(data):
    """Ubah input form menjadi DataFrame dengan urutan kolom sama seperti saat training.

    Preprocessing disamakan dengan notebook Colab: dummy furnishingstatus
    (drop_first) + fitur turunan log_area, bath_x_story, n_facilities.
    """
    row = {c: 0 for c in FEATURE_COLUMNS}
    for c in NUMERIC:
        row[c] = float(data[c])
    for c in YES_NO:
        row[c] = 1 if str(data.get(c, "no")).lower() in ("yes", "ya", "1", "true") else 0

    furnish = str(data.get("furnishingstatus", "furnished")).lower()
    if furnish not in FURNISHING:
        raise ValueError(f"furnishingstatus harus salah satu dari {FURNISHING}")
    row["furnishingstatus_semi-furnished"] = 1 if furnish == "semi-furnished" else 0
    row["furnishingstatus_unfurnished"] = 1 if furnish == "unfurnished" else 0

    # Fitur turunan (sama dengan notebook)
    if row["area"] <= 0:
        raise ValueError("area harus lebih dari 0")
    row["log_area"] = float(np.log(row["area"]))
    row["bath_x_story"] = row["bathrooms"] * row["stories"]
    row["n_facilities"] = sum(row[c] for c in YES_NO)

    return pd.DataFrame([row])[FEATURE_COLUMNS]


@app.route("/")
def index():
    return render_template("index.html", model_ready=model is not None)


@app.route("/health")
def health():
    return jsonify(status="ok", model_loaded=model is not None)


@app.route("/predict", methods=["POST"])
def predict():
    if API_KEY and request.headers.get("X-API-KEY") != API_KEY and request.is_json:
        return jsonify(error="Unauthorized"), 401
    if model is None:
        return jsonify(error="Model belum ditemukan di folder model/."), 500
    data = request.get_json(silent=True) or request.form.to_dict()
    try:
        X = build_features(data)
        pred = float(np.ravel(model.predict(X))[0])
        if LOG_TARGET:
            pred = float(np.expm1(pred))
        return jsonify(predicted_price=round(pred, 2))
    except (KeyError, ValueError) as e:
        return jsonify(error=f"Input tidak valid: {e}"), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)),
            debug=os.environ.get("FLASK_DEBUG", "false").lower() == "true")
