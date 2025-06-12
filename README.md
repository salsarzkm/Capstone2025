# EcoSortAI

**EcoSortAI: Terobosan Cerdas dalam Penyortiran Sampah Otomatis**

EcoSortAI adalah solusi berbasis AI untuk mengotomatisasi penyortiran sampah guna meningkatkan efisiensi pengelolaan limbah dan mendukung keberlanjutan lingkungan. Proyek ini menggunakan model deep learning (VGG16) yang dioptimalkan melalui teknik transfer learning, fine-tuning, dan data augmentasi untuk menghasilkan klasifikasi sampah dengan akurasi yang tinggi.

---

## Isi Repositori

- **Kode Sumber:**  
  Kode Python untuk pengembangan model, yang mencakup penggunaan TensorFlow/Keras, implementasi arsitektur VGG16 beserta transfer learning dan fine-tuning.

- **Notebook Pengembangan:**  
  Dokumen interaktif (Jupyter Notebook) yang mendemonstrasikan langkah-langkah:
  - Preprocessing data (resize, normalisasi)
  - Augmentasi data untuk meningkatkan variasi dan robustnes
  - Pelatihan model dan evaluasi performa (misalnya, confusion matrix dan classification report)

- **README.md:**  
  Panduan lengkap untuk replikasi, yang mencakup:
  - Langkah-langkah instalasi dan konfigurasi lingkungan
  - Instruksi menjalankan kode dan notebook
  - Penjelasan alur kerja serta dependensi yang diperlukan

---

## Fitur Utama

- **Otomatisasi Penyortiran Sampah:**  
  Menggunakan model deep learning berbasis VGG16 untuk mengklasifikasikan sampah ke dalam kategori seperti Anorganik Daur Ulang, Anorganik Tidak Daur Ulang, B3, dan Organik.

- **Transfer Learning & Fine-Tuning:**  
  Memanfaatkan bobot pre-trained dari ImageNet untuk mempercepat pelatihan dan meningkatkan akurasi model melalui penyesuaian lapisan atas (fine-tuning).

- **Data Augmentasi:**  
  Teknik augmentasi (rotasi, zoom, flipping, dan penyesuaian pencahayaan) diterapkan untuk menambah keragaman dataset dan mencegah overfitting.

- **Aplikasi Real-Time:**  
  Model yang telah dilatih diintegrasikan ke dalam aplikasi berbasis Streamlit untuk inferensi dan monitoring secara real-time.

---

## Prasyarat

Pastikan sistem Anda telah terinstall:

- **Python 3.7+**
- **pip** (atau manajer paket lain seperti conda)
- **Virtual Environment** (disarankan: [virtualenv](https://pypi.org/project/virtualenv/) atau [conda](https://docs.conda.io/projects/conda/en/latest/))

---

## Instalasi dan Konfigurasi Lingkungan

1. **Clone Repositori:**

   ```bash
   git clone https://github.com/salsarzkm/EcoSortAI.git
   cd EcoSortAI
   ```

2. **Buat dan Aktifkan Virtual Environment:**

   Menggunakan `virtualenv`:
   
   ```bash
   virtualenv venv
   # Untuk Linux/MacOS:
   source venv/bin/activate
   # Untuk Windows:
   .\venv\Scripts\activate
   ```

3. **Instal Dependencies:**

   Pastikan memiliki file `requirements.txt` yang berisi daftar paket yang diperlukan, kemudian jalankan:

   ```bash
   pip install -r requirements.txt
   ```

   _Contoh file `requirements.txt` dapat berisi:_
   - tensorflow
   - keras
   - numpy
   - pandas
   - matplotlib
   - streamlit
   - scikit-learn
   - dan dependensi lainnya.

---

## Menjalankan Kode dan Notebook

- **Melatih Model:**

  Jalankan skrip pelatihan model:

  ```bash
  python train_model.py
  ```

- **Menjalankan Notebook:**

  Buka notebook pengembangan dengan Jupyter Notebook atau JupyterLab:

  ```bash
  jupyter notebook EcoSortAI_Notebook.ipynb
  ```

- **Menjalankan Aplikasi Streamlit:**

  Untuk melihat demo aplikasi real-time, jalankan:

  ```bash
  streamlit run app.py
  ```

---

## Alur Kerja Proyek

1. **Pengumpulan Data:**  
   Mengumpulkan gambar sampah dari berbagai sumber.

2. **Preprocessing Data:**  
   Resize, normalisasi, dan augmentasi data untuk meningkatkan keragaman gambar.

3. **Pengembangan Model:**
   - **Transfer Learning:** Menggunakan model VGG16 pre-trained.
   - **Fine-Tuning:** Melatih ulang lapisan-lapisan akhir untuk menyesuaikan dengan dataset sampah.
   
4. **Evaluasi Model:**  
   Menghasilkan metrik akurasi, confusion matrix, dan classification report untuk menilai performa.

5. **Deployment:**  
   Mengintegrasikan model ke dalam aplikasi web (Streamlit) untuk inferensi real-time.

---

## Dependensi

- **TensorFlow & Keras :** Untuk pengembangan dan pelatihan model deep learning.
- **OpenCV & PIL:** Untuk pemrosesan dan manipulasi gambar.
- **NumPy & Pandas:** Untuk perhitungan numerik dan pengelolaan data.
- **Matplotlib & Seaborn:** Untuk visualisasi data dan hasil evaluasi.
- **Streamlit:** Untuk deployment aplikasi web.
- **Scikit-Learn:** Untuk evaluasi dan metrik performa.

---

## Dokumentasi Tambahan

- **Diagram Teknis & Flowchart:**  
  Dokumen di folder `docs/` (misalnya, `flowchart.png`) menggambarkan alur proses pengembangan dari pengumpulan data hingga deployment.

- **Panduan Pengguna:**  
  Bagian ini mencakup instruksi replikasi, konfigurasi lingkungan, dan langkah-langkah menjalankan kode, yang telah disediakan dalam file ini.

---
