# Chapter 3 Challenge: Beat the Baseline + Save & Use Your Model ✈️

Studi kasus **Machine Learning dari GDGoC ML Study Jam**. Membangun pipeline klasifikasi end-to-end untuk memprediksi kepuasan penumpang maskapai, mulai dari preprocessing yang anti-data leakage, evaluasi model, perbandingan algoritma, error analysis, hingga menyimpan dan menggunakan model untuk melakukan prediksi.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Zackyalghfr/chapter3-challenge/blob/main/Copy_of_Chapter3_Challenge.ipynb)

## Tentang Dataset

Dataset **Airline Passenger Satisfaction** digunakan untuk memprediksi tingkat kepuasan penumpang berdasarkan berbagai karakteristik pelanggan, perjalanan, serta layanan penerbangan.

Dataset terdiri dari:

* `ps_train.csv` — data training dengan target `satisfaction`
* `ps_test.csv` — data testing tanpa label untuk dilakukan prediksi

Data training memiliki **103.904 data** dengan **24 fitur** dan 1 target.

## Alur Machine Learning

Notebook ini dibagi menjadi beberapa tahap utama:

### 1. Download, Load & Split

Memuat dataset dan memisahkan fitur (`X`) dan target (`y`), kemudian membagi data menjadi training dan validation menggunakan:

* `test_size=0.2`
* `random_state=42`
* `stratify=y`

Penggunaan stratify dilakukan agar proporsi kelas pada data training dan validation tetap seimbang.

### 2. Baseline Model

Membangun baseline menggunakan `DummyClassifier` dengan strategi `most_frequent` sebagai pembanding awal performa model.

Hasil baseline:

* Accuracy: **0.5667**
* Precision: **0.3211**
* Recall: **0.5667**
* F1-score: **0.4099**

### 3. Preprocessing Pipeline

Membangun preprocessing menggunakan `Pipeline` dan `ColumnTransformer` untuk menghindari **data leakage**.

**Numerical features:**

`SimpleImputer(median)` → `StandardScaler`

**Categorical features:**

`SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown="ignore")`

### 4. Model Training & Comparison

Menguji beberapa algoritma klasifikasi:

* Logistic Regression
* Decision Tree
* Random Forest

Setiap model menggunakan preprocessing pipeline yang sama dan dievaluasi menggunakan beberapa classification metrics.

### 5. Error Analysis

Melakukan analisis terhadap data yang salah diprediksi oleh model terbaik untuk mengetahui pola kesalahan dan karakteristik data yang sulit diklasifikasikan.

### 6. Cross Validation

Model terbaik kemudian diuji menggunakan **Stratified K-Fold Cross Validation** dengan 3 folds untuk melihat konsistensi performa model.

### 7. Save & Use Your Model

Model terbaik dilatih kembali menggunakan seluruh data training dan disimpan dalam format `.joblib`.

Model yang telah disimpan kemudian digunakan untuk melakukan prediksi pada `ps_test.csv`.

## Hasil Model

Dari beberapa model yang diuji, **Random Forest** memberikan performa terbaik pada validation set.

| Model               |   Accuracy | F1-weighted |
| ------------------- | ---------: | ----------: |
| Logistic Regression |     0.8765 |      0.8763 |
| Decision Tree       |     0.9459 |      0.9460 |
| **Random Forest**   | **0.9638** |  **0.9637** |

Random Forest juga memperoleh rata-rata **F1-weighted 0.9617** pada Stratified 3-Fold Cross Validation.

## Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib
* Joblib
* Google Colab / Jupyter Notebook

## Output

Model terbaik disimpan sebagai:

```text
best_airline_model.joblib
```

Hasil prediksi pada dataset test disimpan sebagai:

```text
ps_test_predictions.csv
```

## Cara Menjalankan

Klik badge **"Open In Colab"** di atas, atau jalankan secara lokal:

```bash
pip install pandas numpy scikit-learn matplotlib joblib
jupyter notebook Copy_of_Chapter3_Challenge.ipynb
```

Notebook akan mengunduh `ps_train.csv` dan `ps_test.csv` secara otomatis dari repository dataset saat cell download dijalankan.
