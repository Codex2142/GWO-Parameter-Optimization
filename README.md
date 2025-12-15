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