# Jalankan di sel terakhir notebook Colab (setelah model terbaik selesai dilatih).
# Ganti `best_model` dan `X_train` sesuai nama variabel di notebook Anda.
import joblib, json
from google.colab import files

joblib.dump(best_model, "model.pth")          # ekstensi .pth sesuai instruksi tugas
with open("feature_columns.json", "w") as f:
    json.dump(list(X_train.columns), f)       # urutan kolom harus sama dengan saat training

files.download("model.pth")                   # <-- ambil screenshot proses ini
files.download("feature_columns.json")
