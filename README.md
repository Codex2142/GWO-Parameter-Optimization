# Planning
Menggunakan algoritma Gray Wolf Optimizer (GWO) untuk mengoptimasi parameter XGBClassifier (Booster: gbtree).
# Tujuan
- Meningkatkan kinerja model XGBoost menggunakan optimasi berbasis - - - meta-heuristic (GWO).

- Mencari kombinasi parameter paling optimal berdasarkan fungsi objektif misalnya:
    - Logloss
    - AUC
    - Accuracy -> disarankan AUC untuk dataset tidak seimbang.
# parameter yang dituning
```sh
model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    tree_method='hist',
    
    # Parameter yang dioptimasi oleh GWO
    n_estimators=300,       # optimasi
    learning_rate=0.05,    # optimasi
    max_depth=4,           # optimasi
    min_child_weight=1,    # optimasi
    subsample=0.8,         # optimasi
    colsample_bytree=0.8,  # optimasi
    gamma=0,               # optimasi
    reg_alpha=0,           # optimasi
    reg_lambda=1,          # optimasi
)
```
| Parameter |Catatan|
|---|---|
| ```n_estimators ``` | rentang wajar 50–1000 |
| ```learning_rate ``` | rentang 0.001–0.3 |
| ```max_depth ``` | integer 3–12 |
| ```min_child_weight ``` | integer 1–10|
| ```subsample``` | 0.5–1.0|
| ```colsample_bytree ``` | 0.5–1.0|
| ```gamma ``` | 0–5|
| ```reg_alpha ``` | 0–5|
| ```reg_lambda ``` | 0–5|

# Preprocessing
jangan lupa nanti encoding target fitur ya
```sh
y = y.map({'M':1, 'B':0})
```

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