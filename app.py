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
# Ubah ke True jika di Colab target dilatih dengan np.log1p(price)
LOG_TARGET = os.environ.get("LOG_TARGET", "false").lower() == "true"

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


def build_features(data):
    """Ubah input form menjadi DataFrame dengan urutan kolom sama seperti saat training."""
    row = {c: 0 for c in FEATURE_COLUMNS}
    for c in NUMERIC:
        row[c] = float(data[c])
    for c in YES_NO:
        row[c] = 1 if str(data.get(c, "no")).lower() in ("yes", "ya", "1", "true") else 0
    furnish = data.get("furnishingstatus", "furnished")
    for col in ("furnishingstatus_semi-furnished", "furnishingstatus_unfurnished"):
        if col in row:
            row[col] = 1 if col.endswith(furnish) else 0
    return pd.DataFrame([row])[FEATURE_COLUMNS]


@app.route("/")
def index():
    return render_template("index.html", model_ready=model is not None)


@app.route("/health")
def health():
    return jsonify(status="ok", model_loaded=model is not None)


@app.route("/predict", methods=["POST"])
def predict():
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
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
