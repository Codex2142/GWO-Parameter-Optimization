# Cara menjalankan
### Persiapan
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

### Instalasi Lib
```sh
pip install -r requirements.txt
```

### Persiapan Dataset
kemudian anda dapat menjalankan file ```ipynb```

jalankan terlebih dahulu ```preparation.ipynb```  
pilih kernel yang telah diinstall barusan, pastikan namanya adalah ```env (3.10.0)```

### Struktur Direktori
```sh
TUBES/
├── code/
│   ├── xgboost.ipynb         <= metode klasifikasi
│   ├── gwo.py                <= algoritma optimasi
│   └── preparation.ipynb     <= pengunduh dataset
├── dataset/
│   ├── test.data
│   └── train_val.data
├── env/
└── flowchart/

```
sekarang program sudah siap untuk menjalankan program yang diberikan label ```metode klasifikasi```