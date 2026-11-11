# Sistem Prediksi Harga Rumah

Tugas Besar 3 Machine Learning - Triyanti Putri, Teknik Informatika, UNINDRA.
Aplikasi web (Flask) yang memuat model regresi hasil training di Google Colab
dan memperkirakan harga rumah dari karakteristik propertinya.

## Struktur
```
app.py                    # Flask: halaman web + endpoint /predict
templates/index.html      # form input & hasil
model/model.pth           # model dari Colab (letakkan di sini)
model/feature_columns.json
colab_simpan_model.py     # kode untuk menyimpan model di Colab
```

## Menjalankan
```bash
python -m venv venv
venv\Scripts\activate        # Windows  (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
python app.py
```
Buka http://localhost:5000

## Contoh API
```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" \
  -d '{"area":5000,"bedrooms":3,"bathrooms":2,"stories":2,"parking":1,"mainroad":"yes","guestroom":"no","basement":"no","hotwaterheating":"no","airconditioning":"yes","prefarea":"no","furnishingstatus":"furnished"}'
```
Jika model dilatih dengan `np.log1p(price)`, jalankan dengan `LOG_TARGET=true`.
