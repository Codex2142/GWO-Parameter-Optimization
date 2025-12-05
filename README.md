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