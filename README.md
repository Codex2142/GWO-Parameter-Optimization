# Flowchart Sistem
![Flowchart Sistem](flowchart/flowchart.png "alur GWO dan XGBoost")


# yang perlu dicantumkan dalam paper
1. flowchart
2. parameter yang digunakan dan value nya
4. algoritma Swarm fix = ```GWO```
5. visualisasi grafik ```GWO ```tiap iterasi
6. algoritma klasifikasi lebih dari 1 (boleh selain XGBoost)
7. tambahkan metrik evaluasi F1 Score
8. Beda algoritma klasifikasi, maka beda file ```ipynb```
9. penulisan pakai latex (opsional)

# Job Desc Elang & Sandro
1. duplikat kode program ```2-xgboost optimized.ipynb```
2. kemudian pahami flowchart
3. pokoknya ganti metode XGBoost dengan metode lain
4. usahakan parameter GWO harus sama dengan file ```2-xgboost optimized.ipynb``` contohnya nanti : ```3-randomforest optimized.ipynb```
5. tambahkan visualisasi grafik ```GWO``` saat iterasi
6. tampilkan hasil akhir dari ```training model``` (bukan hasil iterasi GWO)
7. Usahakan deadline ```27 Desember```, karena setelah itu penulisan paper

# Optimization of Classification Algorithm Parameters Using Gray Wolf Optimizer approach
dalam projek kali ini, algoritma Gray Wolf Optimizer (GWO) yang termasuk dalam kelompok algoritma Swarm Intelligence berfugnsi untuk memalukan optimasi parameter dari algoritma klasifikasi yang akan diimplementasikan
### Struktur Folder
```sh
tubes/
├── app/
│   ├── classification/
│   │   ├── svm.py
│   │   ├── knn.py
│   │   ├── xgboost.py
│   │   ├── random_forest.py
│   │   └── logistic_regression.py
│   │
│   ├── templates/
│   │   └── index.html
│   │
│   ├── __init__.py
│   ├── config.py
│   ├── gwo.py
│   ├── routes.py
│   └── services.py
│
├── dataset/
│   ├── raw/
│   │   └── dataset.data
│   │   
│   └── cooked/
│       ├── test.data
│       └── train_val.data
│
├── env/
│   ├── (folder lainnya)/
│   └── Script/
│       ├── file lainnya
│       └── activate
│
├── app.py
├── README.md
└── requirements.txt
```

### Cara menjalankan
pastikan anda memiliki python versi ```3.10```  
jalankan perintah ini pada direktori ```tubes```
```sh
env/Scripts/activate
```
jika berhasil, maka terminal akan berubah menjadi seperti dibawah ini
```sh
(env) PS E:\Riwayat Kuliah\Semester 7\Swarm Intelligence\Tubes> 
```
setelah anda menggunakan venv yang telah disediakan, berikutnya install library pendukung program ini dengan cara
> [!CAUTION]
> Proses ini akan memerlukan koneksi internet stabil dan waktu yang cukup lama
```sh
pip install -r requirements.txt
```
setelah melakukan instalasi library yang dibutuhkan, tahap selajutnya adalah menjalankan file ```app.py```
```sh
py app.py
```
setelah itu akan muncul pesan bahwa server sedang berjalan pada port ```5000```
```sh
 * Serving Flask app 'app' (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: on
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 263-507-187
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```

### PR
- grafik iterasi belum
- revisi flowchart bagian ```evaluasi```
- koreksi parameter lagi
- cek apakah tiap algoritma klasifikasi sudah imbang

- algoritma training dan non-training

- skema pengujian (cukup paper / penjealasnan)
- tergantung bahas breast cancer bisa 5 kutip
- traditinonal clssification
- harus penyakit tumor
- brp metode knn,svm
- proposed system , sitem yang diajukan menjelaskan bagan
- methodology mirip bab 2 sempro dan ditambahkan sedikit bab 3
- disarankan formula
- 6 - 10 halaman
- bab 4 dataset ditaruh metodologi sblm proposed method
- introduction kepanjangen
- author dihapus
- result and discussion
- acknowldgement
- references

# Abstrak
Cantumkan secara singkat dan padat:

- Latar belakang masalah (1–2 kalimat)

- Tujuan penelitian

- Metode utama (algoritma klasifikasi + GWO)

- Dataset yang digunakan

- Hasil utama (peningkatan performa)

- Kesimpulan singkat

```Panjang ideal: ±150–250 kata```

# Pendahuluan (Introduction)

### Paragraf 1 – Permasalahan Umum

 - Tingginya angka kanker payudara

- Pentingnya diagnosis dini berbasis data

- Peran machine learning dalam dunia medis

### Paragraf 2 – Permasalahan Spesifik

- Keterbatasan performa algoritma klasifikasi standar

- Sensitivitas algoritma terhadap pemilihan parameter

### Paragraf 3 – Literatur Algoritma Klasifikasi

- Pemanfaatan algoritma klasifikasi dalam diagnosis kanker

- Contoh algoritma yang sering digunakan (SVM, KNN, Random Forest, dll.)

### Paragraf 4 – Literatur GWO dalam Optimasi

- Konsep optimasi metaheuristik

- GWO sebagai metode optimasi parameter

- Keunggulan GWO dibanding optimasi konvensional

### Paragraf 5 – Tujuan dan Kontribusi Penelitian

- Tujuan penelitian

- Kontribusi utama:

  - Penerapan GWO untuk optimasi parameter

  - Perbandingan performa sebelum dan sesudah optimasi

  - Evaluasi pada dataset kanker payudara

# Penelitian Terkait (Related Work)
Cantumkan:

- Minimal 5 penelitian terdahulu

- Fokus pada:

  - GWO untuk optimasi parameter

  - Perbandingan classifier tanpa optimasi vs dengan optimasi

- Sertakan:

  - Algoritma klasifikasi

  - Metode optimasi

  - Dataset

  - Hasil evaluasi

kalo bisa pakai tabel dibawha ini:

| Author | Year | Classifier | Optimization | Dataset | Accuracy | Precision | F-1 Score |
| ------ | ---- | ---------- | ------------ | ------- | -------- | --------  | --------  |


# Metodologi Penelitian (Methodology)

### A. Kanker Payudara

- Definisi kanker payudara

- Dampak dan pentingnya diagnosis dini

- Hubungan dengan analisis data medis

### B. Algoritma Klasifikasi

- Pengertian algoritma klasifikasi

- Taksonomi algoritma klasifikasi

- Kelompok algoritma:

- Linear

- Probabilistik

- Tree-based

- Instance-based

- Sertakan gambar taksonomi algoritma klasifikasi

- Contoh algoritma + sitasi

### C. Grey Wolf Optimizer (GWO)

- Inspirasi perilaku serigala abu-abu

- Struktur hierarki (α, β, δ, ω)

- Mekanisme perburuan

- Persamaan matematis utama

- Peran GWO dalam optimasi parameter classifier

### D. Kerangka Kerja Sistem (Proposed Framework)

- Alur kerja penelitian:

- Input dataset

- Preprocessing

- Training classifier

- Optimasi parameter dengan GWO

- Evaluasi performa

# Deskripsi Dataset (Dataset Description)

- Cantumkan:

  - Nama dataset

  - Sumber dataset

  - Jumlah data

  - Jumlah kelas

  - Distribusi kelas

- Fitur Dataset

  - Daftar fitur

  - Jenis fitur:

    - Numerik (desimal)

    - Biner

    - Kategorikal

- Fitur yang digunakan dalam penelitian

  - Disarankan menggunakan tabel deskripsi fitur


# Hasil dan Pembahasan (Results and Discussion)
- Hasil Eksperimen

- Performa algoritma tanpa GWO

- Performa algoritma dengan GWO

- Metrik evaluasi:

  - Accuracy

  - Precision

  - Recall

  - F1-score

- Analisis dan Diskusi

  - Perbandingan hasil

  - Dampak optimasi GWO

  - Penjelasan peningkatan atau penurunan performa

  - Kelebihan dan keterbatasan metode

# Kesimpulan (Conclusion)
Cantumkan:

- Ringkasan hasil penelitian

- Jawaban terhadap tujuan penelitian

- Implikasi hasil penelitian

- Saran pengembangan penelitian selanjutnya

# Acknowledgement

- Institusi

- Dosen pembimbing

- Sumber pendanaan (jika ada)