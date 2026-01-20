Pendahuluan
QR Code Security Analyzer adalah aplikasi berbasis web yang dibangun menggunakan Streamlit untuk menganalisis dan membandingkan performa model Deep Learning dalam mendeteksi ancaman Quishing (QR Phishing). Aplikasi ini berfungsi sebagai platform evaluasi komprehensif bagi para peneliti dan data ilmuwan untuk menelaah efektivitas arsitektur CNN (MobileNetV2), LSTM, dan GRU.

Latar Belakang
Proyek ini dikembangkan untuk memenuhi evaluasi Ujian Akhir Semester (UAS) Pembelajaran Mendalam. Fokus utama adalah memitigasi risiko keamanan dari QR Code yang secara visual tidak transparan (opaque), sehingga sering dimanipulasi untuk menyebarkan tautan berbahaya (malicious).

Fitur
1. Manajemen Dataset (Dataset Management)
  - Unggah dataset dalam format ZIP yang berisi citra QR Code.
  - Deteksi otomatis folder benign (aman) dan malicious (berbahaya).
  - Visualisasi statistik dataset secara real-time.
  - Preview sampel citra beserta metadata yang diekstraksi.

2. Analisis Model (Model Analysis)
  - Mendukung visualisasi hasil pelatihan dari arsitektur CNN, LSTM, dan GRU.
  - Metrik performa detail: Akurasi, Presisi, Recall, dan F1-Score.
  - Visualisasi riwayat pelatihan (training history) untuk mendeteksi overfitting.

3. Fitur Unik 
  - Inovasi 1 (Global MaxPooling 1D): Peningkatan ekstraksi fitur sekuensial pada model LSTM/GRU untuk menangkap sinyal ancaman paling dominan pada URL dengan   menggunakan Mobile NetV2 untuk proses training CNN.
  - Inovasi 2 (Hyperparameter Tuning): Hasil komparasi didasarkan pada optimasi parameter sistematis (seperti learning rate dan dropout) untuk mendapatkan model paling akurat.

4. Visualisasi Lanjutan 
  - Tabel perbandingan performa model secara head-to-head.
  - Grafik interaktif untuk distribusi dataset.
  - Radar charts dan bar charts untuk membandingkan karakteristik antar arsitektur.

5. Generasi Laporan (Report Generation)
  - Ekspor hasil analisis ke format JSON, CSV, atau teks.
  - Unduh laporan komparatif model secara otomatis.

6. Teknologi Deployment
   Aplikasi ini merepresentasikan transisi fungsional dari analisis laboratorium (Google Colab) menuju sistem siap pakai yang berdampak nyata.
  - Bahasa Pemrograman: Python
  - Kerangka Kerja Web: Streamlit
  - Deep Learning Libraries: TensorFlow/Keras
  - Pengolah Citra: OpenCV
  - Deployment: Platform Web Streamlit (Aksesibilitas Real-Time)

7. Cara Penggunaan
  1. Siapkan Dataset: Pastikan dataset ZIP memiliki struktur folder /benign dan /malicious. (contohnya seperti folder dataset/1000 QR Images of Malicious and Benign QR codes 2025)
  2. Jalankan Aplikasi: Unggah ZIP melalui menu Dataset ZIP.
  3. Analisis: Masukkan data hasil pelatihan model dengan ketentuan eval.json dan history.json untuk melihat perbandingan grafik metrik secara otomatis. (contohnya seperti models/cnn_eval.json)
  4. Ekspor: Unduh laporan ringkasan untuk kebutuhan dokumentasi ilmiah. 