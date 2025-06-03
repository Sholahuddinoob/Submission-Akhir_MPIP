# Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan – Prediksi Dropout Mahasiswa

## 1. Business Understanding

Sebuah institusi pendidikan tinggi mengalami tantangan dalam mempertahankan mahasiswa hingga kelulusan. Berdasarkan data historis, terdapat cukup banyak mahasiswa yang mengundurkan diri (dropout) atau tidak melanjutkan studi. 

Manajemen perguruan tinggi membutuhkan sistem cerdas berbasis data untuk:
- Mengidentifikasi mahasiswa yang berpotensi dropout lebih awal
- Memahami faktor-faktor penyebab utama
- Membantu penyusunan strategi intervensi akademik

## 2. Permasalahan Bisnis

1. Tingginya angka dropout mahasiswa, yang berdampak pada kualitas dan reputasi institusi.
2. Tidak adanya sistem visual untuk memantau kinerja dan risiko dropout secara berkala.
3. Belum tersedia alat bantu prediksi status mahasiswa berdasarkan data awal dan akademik.

## 3. Cakupan Proyek

- Membersihkan dan mempersiapkan data mahasiswa (data preparation).
- Melakukan Exploratory Data Analysis (EDA) untuk menemukan faktor-faktor risiko.
- Membangun model machine learning (Random Forest) untuk memprediksi status akhir mahasiswa.
- Membuat prototipe prediksi interaktif dengan Streamlit.
- Membangun dashboard visualisasi edukatif menggunakan Metabase.

## 4. Persiapan

### Sumber Data

Dataset digunakan dari Dicoding:  
https://github.com/dicodingacademy/dicoding_dataset/tree/main/student

- Jenis: Dataset fiktif mahasiswa
- Jumlah baris: ~4.000 (setelah preprocessing)

### Tools & Environment

Python 3.12.3  
pandas==1.5.3  
numpy==1.21.6  
matplotlib==3.5.3  
seaborn==0.12.2  
scikit-learn==1.2.2  
imbalanced-learn==0.10.1  
joblib==1.2.0  
streamlit==1.22.0  
Metabase (via Docker)

### Setup Environment

Menggunakan Anaconda:
- conda create --name student-dropout python=3.9
- conda activate student-dropout
- pip install -r requirements.txt

Menggunakan Shell/Terminal:
- pip install pipenv
- pipenv install
- pipenv shell
- pip install -r requirements.txt

Cara Instalasi Umum:
- pip install -r requirements.txt

### Menjalankan Aplikasi Prediksi

- streamlit run app.py

File ini akan memuat model .pkl, menerima input dari user, dan memberikan prediksi status mahasiswa (Dropout, Enrolled, Graduate).

## 5. Business Dashboard

Dashboard dibangun menggunakan Metabase, dengan beberapa visualisasi utama untuk mendukung pengambilan keputusan:

- Distribusi Status Mahasiswa: Menunjukkan proporsi mahasiswa dropout, lulus, dan masih aktif.
- Status vs Gender: Menggambarkan distribusi status berdasarkan jenis kelamin.
- Nilai Akademik vs Status: Menunjukkan hubungan antara rata-rata nilai semester dan kemungkinan dropout.
- Umur Masuk vs Status: Memberikan insight apakah umur saat pendaftaran berpengaruh terhadap keberhasilan studi.

### Informasi Login Metabase (Opsional)

- Email: root@mail.com  
- Password: root123  

- File database Metabase: metabase.db.mv.db  
- Screenshot dashboard: sholahuddin-dashboard.png  

## 6. Conclusion

- Mahasiswa dengan nilai akademik semester rendah, usia masuk lebih tua, dan status keuangan bermasalah (debtor) memiliki kemungkinan lebih tinggi untuk dropout.
- Model Random Forest memberikan akurasi yang baik (~77%) dalam klasifikasi multi-kelas (Graduate, Dropout, Enrolled).
- Aplikasi Streamlit dapat digunakan staf kampus untuk prediksi status mahasiswa secara individual dengan mudah.

## 7. Rekomendasi (Optional)

- Lakukan intervensi dini terhadap mahasiswa yang menunjukkan kombinasi risiko:
  - Nilai < 10
  - Debtor = 1
  - Usia > 22
- Tawarkan program mentoring atau beasiswa untuk mahasiswa dengan risiko akademik.
- Integrasikan dashboard Metabase ke dalam rapat evaluasi akademik rutin.
- Pertimbangkan melatih ulang model dengan data terkini dan teknik balancing data.
