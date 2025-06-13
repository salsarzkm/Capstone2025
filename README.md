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

## **Business Understanding**

### **Pernyataan Masalah**

Saat ini, pengelolaan sampah di banyak kota mengalami beberapa kendala kritis yang berdampak pada efisiensi operasional dan kualitas lingkungan, yaitu:
- Proses Pemilahan yang Manual dan Lambat:
Metode konvensional untuk memilah sampah dilakukan secara manual, sehingga rawan terjadi kesalahan dan sangat tidak efisien untuk menangani volume sampah yang terus meningkat.
- Ketidakakuratan dalam Klasifikasi Sampah:
Tanpa adanya sistem otomatis, identifikasi jenis sampah (misalnya sampah anorganik, berbahaya, elektronik, organik, dan yang dapat didaur ulang) kerap mengalami inkonsistensi. Hal ini tidak hanya menghambat proses daur ulang, tetapi juga berpotensi menimbulkan risiko kesehatan dan lingkungan.
- Keterbatasan Data untuk Pengambilan Keputusan:
Minimnya data yang terintegrasi dan terkelola dengan baik menyebabkan sulitnya menganalisa arus dan komposisi sampah, sehingga pengambil kebijakan tidak mendapatkan insight yang diperlukan untuk membuat perbaikan strategis.


### **Tujuan**

Tujuan utama dari proyek ini adalah untuk mengembangkan sebuah aplikasi yang mengintegrasikan teknologi machine learning guna mengatasi masalah pengelolaan sampah secara menyeluruh. Adapun tujuan spesifiknya meliputi:
- Otomatisasi Klasifikasi Sampah:
Membangun model machine learning berbasis TensorFlow yang mampu secara otomatis mengklasifikasikan jenis sampah dari data citra dengan akurasi tinggi.
- Meningkatkan Efisiensi Operasional:
Mengubah proses pemilahan dari metode manual ke sistem otomatis sehingga dapat mengurangi waktu, tenaga, dan potensi kesalahan dalam identifikasi.
- Menyediakan Data Insight untuk Pengambilan Keputusan:
Menghasilkan data yang terintegrasi dari proses klasifikasi untuk mendukung analisis lebih lanjut, yang nantinya dapat digunakan oleh pihak berwenang dalam merancang strategi pengelolaan sampah yang lebih adaptif dan efisien.

### **Solution Approach**

1. **Pendekatan Utama (Custom CNN Model dengan TensorFlow)**
   - **Pengembangan Model:**
     Membangun model Convolutional Neural Network (CNN) dari awal menggunakan TensorFlow. Pendekatan ini dilakukan tanpa menggunakan model dari TensorFlow Hub atau resource serupa, sesuai dengan ketentuan.
   - ***Preprocessing* dan Augmentasi Data:**
     Memastikan data citra sampah yang dikumpulkan dan disimpan di Google Drive diproses secara optimal melalui tahap normalisasi, resizing, dan augmentasi untuk meningkatkan keragaman dan robustness model dalam berbagai kondisi nyata.
   - **Inferensi Sederhana:**
     Menyusun kode inferensi yang sederhana untuk mengujicobakan model dalam mengidentifikasi jenis sampah secara real-time, sehingga aplikasi dapat dengan mudah diintegrasikan dengan antarmuka.
     
2. **Pendekatan Alternatif (Benchmark dengan Transfer Learning – Opsional/Side Quest):**
   - **Eksperimen Model Transfer Learning:**
     Sebagai nilai tambah (opsional), proyek dapat dilengkapi dengan model tambahan yang menggunakan transfer learning dari CNN pre-trained (misalnya, MobileNetV2) untuk menjadi benchmark evaluasi performa.
   - **Perbandingan Kinerja:**
     Dengan membandingkan model custom dengan model transfer learning, akan diperoleh insight mengenai kelebihan dan kekurangan dari masing-masing pendekatan, sehingga dapat menjadi dasar perbaikan untuk implementasi di masa mendatang.
   - **Integrasi dan Deployment:**
     Walaupun penggunaan model transfer learning diperbolehkan hanya sebagai side quest, integrasinya ke dalam aplikasi end-to-end dapat memberikan nilai tambah melalui fitur-fitur seperti pilihan mode inferensi dan optimasi performa.

Dengan kombinasi pendekatan utama dan alternatif tersebut, proyek ini bertujuan untuk memberikan solusi yang tidak hanya memenuhi kebutuhan dasar dalam pengelolaan sampah, tetapi juga membuka peluang untuk pengembangan lebih lanjut melalui eksperimen teknologi canggih. Proses ini akan memastikan bahwa aplikasi yang dibangun mampu dioperasikan secara andal dan memberikan dampak positif pada tata kelola lingkungan serta pengambilan keputusan strategis.

---

## **Data Understanding**

### **1. Sumber Data**

Proyek ini menggunakan dua sumber dataset eksternal yang telah diunduh dari Kaggle:
1. **Dataset 1: Recyclable and Household Waste Classification**
    - Tautan: [Recyclable and Household Waste Classification](https://www.kaggle.com/datasets/joebeachcapital/realwaste)
    - Deskripsi Umum:
      Dataset ini merupakan kumpulan citra berkualitas tinggi yang menggambarkan sampah rumah tangga serta bahan-bahan yang dapat didaur ulang. Data diorganisasikan dalam struktur folder, di mana setiap folder mewakili kategori label. Berdasarkan deskripsi dan sumber terkait, dataset ini diperkirakan terdiri dari sekitar 15.000 gambar dan mencakup ratusan kategori (misalnya, plastik, kertas, kaca, logam, organik, dan lain-lain).
    - Format File: Gambar dalam format JPG atau PNG dengan resolusi yang konsisten (misalnya, 256×256 piksel).

2. **Dataset 2: RealWaste Dataset**
    - Tautan: [Real Waste](https://www.kaggle.com/datasets/joebeachcapital/realwaste)
    - Deskripsi Umum:
      Dataset ini merupakan kumpulan citra nyata yang diambil di lingkungan landfill (tempat pembuangan akhir). Dataset RealWaste terdiri dari 4.752 gambar yang telah dikumpulkan secara autentik. Setiap gambar telah diberi label berdasarkan material sampah, sehingga menyediakan informasi yang mendalam untuk klasifikasi.
    - Rincian Label dan Jumlah Gambar:
      * Cardboard: 461
      * Food Organics: 411
      * Glass: 420
      * Metal: 790
      * Miscellaneous Trash: 495
      * Paper: 500
      * Plastic: 921
      * Textile Trash: 318
      * Vegetation: 436
    - Format File:
      Gambar berwarna (RGB) dalam resolusi sekitar 524×524 piksel atau resolusi serupa sesuai dengan standar yang diterapkan oleh pengumpul data.
      
### **2. Pembentukan Dataset**

- Dataset yang digunakan dalam proyek ini dihasilkan dengan mengintegrasi dua sumber data eksternal dari Kaggle tersebut.
- Setelah proses pengumpulan, pembersihan, dan penyelarasan, kedua dataset tersebut dikombinasikan dan direstrukturisasi secara mandiri oleh tim untuk menghasilkan satu dataset final yang konsisten dengan kebutuhan proyek. Dataset final ini kemudian dibagi menjadi 5 kategori utama sesuai dengan fokus proyek, yaitu:
  * Sampah Anorganik
  * Sampah Berbahaya
  * Sampah Elektronik
  * Sampah Organik
  * Sampah yang Bisa Didaur Ulang
- Dataset hasil integrasi dan pembagian kategori ini telah disimpan secara terpusat dan dapat diakses melalui Google Drive: [Dataset Capstone Project](https://drive.google.com/drive/folders/1iWAHYIqiK6B8bj5YJqCr98gAF50-hA4S?usp=drive_link)

### **3. Deskripsi Fitur dan Variabel**

Pada dataset final yang telah disusun, informasi utama yang terdapat di dalamnya adalah
- **Gambar (Citra):**
Masing-masing instance berupa gambar berformat JPEG atau PNG yang mengilustrasikan berbagai jenis sampah. Gambar memiliki variasi resolusi dan kondisi pencahayaan, mengingat sebagian data diambil dalam kondisi studio (Dataset 1) dan sebagian lagi dalam kondisi nyata (Dataset 2).
- **Label/Kategori:**
Setiap gambar diberi label sesuai dengan salah satu dari 5 kategori:
* Sampah Anorganik: Meliputi sampah yang umumnya berupa bahan non-organik seperti plastik, logam, dan kaca.
* Sampah Berbahaya: Sampah yang mengandung bahan kimia atau komponen beracun.
* Sampah Elektronik: Mengacu pada sampah dari peralatan elektronik yang sudah tidak terpakai.
* Sampah Organik: Berupa sisa-sisa bahan organik seperti sisa makanan dan dedaunan.
* Sampah yang Bisa Didaur Ulang: Gambar-gambar yang menunjukkan bahan-bahan yang bisa diolah ulang menjadi produk baru.
  
- **Struktur Data:**
Data diorganisasikan dalam struktur folder, di mana folder utama telah dibagi menjadi 5 subfolder berdasarkan kategori yang telah ditetapkan. Ini memudahkan proses automatisasi data (misalnya melalui fungsi `tf.keras.preprocessing.image_dataset_from_directory`) dan analisis lebih lanjut.

---

## Isi Repositori

- **Kode Sumber:**  
  Kode Python untuk pengembangan model, yang mencakup penggunaan TensorFlow/Keras, implementasi arsitektur VGG16 beserta transfer learning dan fine-tuning.

- **Notebook Pengembangan:**  
  Dokumen interaktif (Jupyter Notebook) yang mendemonstrasikan langkah-langkah:
  - Split Data
    #### Split ke folder train, val, test
    splitfolders.ratio(
        original_dataset_dir,
        output="/content/dataset_split",
        seed=42,
        ratio=(0.7, 0.15, 0.15),
        move=False  # False = copy file, True = pindahkan file
    )
    
    #### Path setelah split
    train_dir = '/content/dataset_split/train'
    val_dir = '/content/dataset_split/val'
    test_dir = '/content/dataset_split/test'
    
  - Preprocessing data (resize, normalisasi) & Augmentasi data untuk meningkatkan variasi dan robustnes
    #### Augmentasi hanya untuk training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        zoom_range=0.3,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3]
    )
    
    #### Untuk validation & test (hanya rescale)
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    #### Generator
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False  # Penting untuk evaluasi akurasi per kelas nanti
    )
    
  - Pelatihan model
    #### 1. Load VGG16 tanpa fully connected layer, freeze semua layer dulu
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False
    
    #### 2. Tambah top layers
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(4, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    #### 3. Compile model dengan learning rate agak besar untuk training awal
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

  - Evaluasi performa (confusion matrix dan classification report)
    #### 1. Prediksi data test
    y_pred_probs = model.predict(test_generator)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    #### 2. Label asli dari generator
    y_true = test_generator.classes
    
    #### 3. Nama kelas (urutan sesuai class_indices)
    class_names = list(test_generator.class_indices.keys())
    
    #### 4. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    #### 5. Visualisasi Confusion Matrix
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Klasifikasi Sampah')
    plt.show()
    
    #### 6. Classification Report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
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
