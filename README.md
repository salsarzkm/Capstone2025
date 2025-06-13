---
## **Project Overview (Ulasan Proyek)**

### **Latar Belakang**

Di tengah pertumbuhan populasi dan urbanisasi yang pesat, pengelolaan sampah telah menjadi salah satu tantangan terbesar di banyak negara, khususnya di Indonesia. Volume sampah yang terus meningkat tidak hanya memberi tekanan pada infrastruktur pengelolaan, tetapi juga menimbulkan dampak serius terhadap lingkungan dan kesehatan masyarakat. Saat ini, proses pemilahan dan pengelolaan sampah masih banyak dilakukan secara manual, yang rentan terhadap kesalahan dan kurang efisien.

Dalam situasi ini, teknologi **Machine Learning** menawarkan solusi yang inovatif dan praktis. Dengan menganalisis data citra secara otomatis, model machine learning dapat mengidentifikasi jenis-jenis sampah seperti Sampah Anorganik, Sampah Berbahaya, Sampah Elektronik, Sampah Organik, dan Sampah yang Bisa Didaur Ulang. Implementasi model berdasarkan TensorFlow yang dikembangkan dari awal ini diharapkan mampu meningkatkan akurasi dan kecepatan identifikasi, sehingga membantu meningkatkan efisiensi pengelolaan dan daur ulang sampah.

### Urgensi Proyek

Proyek ini sangat penting karena:
- **Otomatisasi Identifikasi:** Mengurangi ketergantungan pada proses manual yang lambat dan rawan kesalahan dengan menerapkan sistem otomatisasi yang mampu mengklasifikasikan sampah secara real-time.
- **Efisiensi Operasional:** Mempercepat proses pemilahan sampah yang berujung pada pengelolaan yang lebih efisien dan mengurangi biaya operasional dalam proses pengolahan sampah.
- **Dukungan Terhadap Lingkungan:** Mengoptimalkan daur ulang dan pemrosesan sampah yang ramah lingkungan, sehingga berkontribusi pada pencegahan pencemaran dan peningkatan kualitas hidup masyarakat.
- **Penerapan Teknologi Lokal:** Memberikan peluang untuk mengembangkan solusi teknologi lokal yang relevan dengan konteks dan tantangan geografis, ekonomi, dan sosial di Indonesia.

### **Riset dan Referensi**

Berbagai penelitian telah menunjukkan bahwa pendekatan otomatis menggunakan machine learning dan computer vision dapat mengatasi tantangan ini dengan memungkinkan identifikasi sampah secara real-time serta memberikan data akurat untuk proses daur ulang dan pengolahan limbah. Misalnya, studi yang dipublikasikan dalam "Predictive Analytics In Waste Management: Harnessing Machine Learning For Sustainable Solutions" menyoroti potensi analitik prediktif dalam mengoptimalkan sistem pengelolaan sampah secara berkelanjutan dengan mengintegrasikan teknologi machine learning dan sensor canggih .

Sebuah survei mendalam yang dipublikasikan oleh MDPI dalam artikel "Solid Waste Generation and Disposal Using Machine Learning Approaches: A Survey of Solutions and Challenges" menguraikan tantangan serta solusi berbasis machine learning yang dapat meningkatkan keakuratan prediksi dan efisiensi operasional dalam pengelolaan sampah. Selain itu, ulasan terbaru dalam "Artificial Intelligence for Waste Management in Smart Cities: A Review" dari Springer menunjukkan bahwa penerapan kecerdasan buatan tidak hanya mengurangi biaya logistik tetapi juga mengoptimalkan proses pemilahan dan daur ulang sampah, yang merupakan langkah penting dalam transformasi menuju smart cities .

#### **Relevansi dan Dampak**

Pengembangan aplikasi cerdas yang mengintegrasikan model machine learning untuk identifikasi sampah ini memiliki dampak strategis dalam mendukung program pengelolaan sampah yang lebih efisien, mengurangi pencemaran, serta memberikan kontribusi nyata terhadap pencapaian target pembangunan berkelanjutan. Dengan mengadopsi pendekatan teknologi ini, diharapkan dapat meningkatkan akurasi pemilahan sampah dan memberikan data insight yang berguna bagi pengambil kebijakan untuk merancang sistem pengelolaan limbah yang lebih responsif dan adaptif terhadap kondisi riil. Proyek ini tidak hanya menawarkan solusi praktis untuk permasalahan lingkungan yang mendesak, tetapi juga membuka peluang inovasi dalam integrasi teknologi kecerdasan buatan untuk mengoptimalkan proses operasional di berbagai sektor .

**Referensi:**

Beberapa penelitian telah menggarisbawahi potensi besar penggunaan machine learning dalam pengelolaan sampah:

- **Predictive Analytics In Waste Management: Harnessing Machine Learning For Sustainable Solutions**  
  Studi ini mengilustrasikan bagaimana analitik prediktif dan machine learning dapat diintegrasikan dengan teknologi sensor untuk mengoptimalkan pemilahan dan pengelolaan sampah secara berkelanjutan.  
  [Baca selengkapnya](https://ijcrt.org/papers/IJCRT2503714.pdf)

- **Solid Waste Generation and Disposal Using Machine Learning Approaches: A Survey of Solutions and Challenges**  
  Artikel survei yang diterbitkan oleh MDPI ini mengevaluasi berbagai pendekatan machine learning yang telah diaplikasikan dalam mengatasi tantangan pengelolaan sampah, mulai dari prediksi jumlah sampah hingga optimisasi proses pengumpulan.  
  [Baca selengkapnya](https://www.mdpi.com/2071-1050/14/20/13578)

- **Artificial Intelligence for Waste Management in Smart Cities: A Review**  
  Ulasan dari Springer ini memaparkan berbagai aplikasi kecerdasan buatan dalam pengelolaan sampah, termasuk penggunaan AI untuk optimasi rute pengumpulan dan peningkatan efisiensi operasional di era smart cities.  
  [Baca selengkapnya](https://link.springer.com/article/10.1007/s10311-023-01604-3)


Dengan mengacu pada riset dan referensi tersebut, proyek ini diharapkan tidak hanya memberikan solusi praktis dalam identifikasi sampah, tetapi juga mendukung pengembangan strategi pengelolaan sampah yang berkelanjutan dan efisien. Ini merupakan langkah integratif yang mempertemukan inovasi teknologi dengan tantangan pengelolaan lingkungan, sejalan dengan upaya membangun smart cities yang lebih bersih dan pintar.

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

- **Aplikasi Streamlit:**
  - **app.py**: Berisi kode aplikasi Streamlit untuk mengimplementasikan EcoSortAI dalam antarmuka web interaktif.
  - **requirements.txt**: Daftar semua dependensi yang diperlukan untuk menjalankan aplikasi Streamlit.

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
