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

- ## Kode Sumber

Kode sumber proyek ini ditulis dalam bahasa Python untuk mengembangkan model deep learning guna mengotomatisasi proses penyortiran sampah. Secara spesifik, proyek ini mencakup:

- **Penggunaan TensorFlow/Keras:**  
  Menggunakan framework TensorFlow dan Keras untuk membangun, melatih, dan mengevaluasi model. Framework ini menyediakan antarmuka modular yang memudahkan eksperimen dan integrasi dengan komponen lain.

```
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, 
    MaxPooling2D, 
    GlobalAveragePooling2D,
    Dense, 
    Flatten, 
    Dropout, 
    BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
```

- **Implementasi Arsitektur VGG16:**  
  Model VGG16, yang sudah dilatih sebelumnya (pre-trained) pada dataset ImageNet, digunakan sebagai basis. VGG16 berperan dalam ekstraksi fitur visual dari gambar sampah dengan performa yang stabil.

```
# 1. Load VGG16 tanpa fully connected layer, freeze semua layer dulu
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

# 2. Tambah top layers
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
```

- **Transfer Learning:**  
  Proyek memanfaatkan transfer learning dengan memanfaatkan bobot-bobot dari model VGG16 yang telah terlatih. Dengan demikian, model langsung mendapatkan fitur dasar dari dataset besar sebelum diadaptasi ke dataset spesifik pengelolaan sampah, yang secara signifikan mempercepat proses pelatihan dan meningkatkan akurasi. Aspek *transfer learning* ini diimplementasikan melalui penggunaan `base_model.trainable = False` pada tahap awal pelatihan.
  
- **Fine-Tuning:**  
  Setelah memanfaatkan bobot pre-trained, lapisan-lapisan akhir dari VGG16 di-fine-tune agar sesuai dengan karakteristik gambar sampah. Proses inti fine-tuning meliputi:
  - **Pembekuan lapisan awal:** Menjaga fitur-fitur dasar yang sudah dipelajari.
  - **Pelatihan ulang lapisan akhir:** Melatih ulang lapisan-lapisan terakhir untuk menangkap ciri-ciri unik dari dataset sampah (seperti variasi jenis, tekstur, dan warna).
    
```
    # Fine-tuning: unfreeze beberapa layer terakhir VGG16 untuk training lanjut
    for layer in base_model.layers[-4:]: # Unfreeze 4 layer terakhir
        layer.trainable = True

    # Compile ulang dengan learning rate kecil untuk fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Training fine-tuning
    history_finetune = model.fit(
        train_generator,
        epochs=30,
        validation_data=val_generator,
        callbacks=callbacks
    )
```

Pastikan untuk menginstal semua dependensi yang tercantum di dalam `requirements.txt` sebelum menjalankan skrip pelatihan dan evaluasi model.


- **Notebook Pengembangan:**

Dokumen interaktif untuk proyek EcoSortAI ini dijalankan melalui Google Colab. Notebook ini mendemonstrasikan langkah-langkah utama dalam pengembangan dan evaluasi model, sehingga memudahkan pengguna untuk memahami dan mereplikasi proses yang telah dilakukan. Berikut adalah langkah-langkah yang tercakup dalam notebook:

- **Preprocessing Data:**
  - **Resize:** Mengubah ukuran gambar agar konsisten dan sesuai dengan dimensi yang diperlukan sebagai input model.
  - **Normalisasi:** Melakukan penyesuaian nilai piksel agar data berada pada skala yang sama, yang membantu dalam proses konvergensi selama pelatihan.
 
```
    # Untuk validation & test (hanya rescale)
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    # Generator untuk validation & test
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
```

- **Augmentasi Data:**
  - Menerapkan teknik augmentasi seperti rotasi, flipping, zoom, dan penyesuaian pencahayaan untuk meningkatkan variasi dataset.
  - Teknik ini dirancang agar model dapat belajar lebih robust dari berbagai kondisi gambar, sekaligus mengurangi risiko overfitting.
 
```
    # Augmentasi hanya untuk training
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

    # Generator untuk training
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )
```

- **Pelatihan Model dan Evaluasi Performa:**
  - **Pelatihan Model:** Menggunakan TensorFlow/Keras dengan implementasi arsitektur VGG16 yang telah dioptimalkan melalui transfer learning dan fine-tuning, sehingga model dapat mempelajari fitur khusus dari gambar sampah.

 ```
    # Kompilasi model dengan learning rate agak besar untuk training awal
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint("best_vgg16_model.h5", monitor='val_accuracy', save_best_only=True)
    ]

    # Training awal (feature extraction)
    history = model.fit(
        train_generator,
        epochs=20,
        validation_data=val_generator,
        callbacks=callbacks
    )
 ```

  - **Evaluasi:** Menghasilkan visualisasi seperti confusion matrix dan classification report yang menampilkan metrik performa (misalnya, precision, recall, F1-score) untuk setiap kategori, sehingga memudahkan analisis kinerja model.
    
```
    # Evaluasi di test set
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Akurasi pada Test Set setelah Fine-Tuning: {test_acc * 100:.2f}%")

    # Prediksi data test
    y_pred_probs = model.predict(test_generator)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Label asli dari generator
    y_true = test_generator.classes

    # Nama kelas (urutan sesuai class_indices)
    class_names = list(test_generator.class_indices.keys())

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    # Visualisasi Confusion Matrix
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Klasifikasi Sampah')
    plt.show()

    # Classification Report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
```

---

- **Aplikasi Streamlit:**

- [Streamlit](https://ecosortai.streamlit.app/) 
  Aplikasi Streamlit yang mengimplementasikan EcoSortAI dalam antarmuka web interaktif. Mencakup:
  - Pengambilan input gambar dari pengguna (melalui kamera atau upload file).
  - Pemanggilan model deep learning yang telah dilatih (menggunakan VGG16 dengan transfer learning dan fine-tuning) untuk melakukan inferensi secara real-time.
  - Penampilan hasil klasifikasi sampah (misalnya, kategori seperti Anorganik Daur Ulang, Anorganik Tidak Daur Ulang, B3, dan Organik) melalui dashboard interaktif.
  - Visualisasi data pendukung dan metrik performa yang membantu pengguna memahami kinerja model.
  
- **requirements.txt**  
  File `requirements.txt` berisi daftar semua dependensi yang diperlukan untuk menjalankan aplikasi Streamlit. Pastikan Anda menginstal semua paket pada file ini agar aplikasi dapat berjalan dengan lancar. Dependensi yang tercantum antara lain:
  - streamlit
  - tensorflow
  - gdown
  - scikit
  - numpy
  - pandas
  - dan dependensi lainnya

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

   _file `requirements.txt` berisi:_
   - tensorflow
   - split-folders
   - matplotlib
   - seaborn
   - numpy
   - pillow
   - tqdm
   - scikit-learn

---

## Menjalankan Kode dan Notebook

- **Melatih Model:**

## Pengembangan Model Menggunakan VGG16

Proyek EcoSortAI memanfaatkan model VGG16 yang telah dilatih sebelumnya (pre-trained) untuk ekstraksi fitur dari gambar sampah. Proses pengembangan model kami terdiri dari beberapa tahap, yaitu:

1. **Load VGG16 tanpa Fully Connected Layer & Freeze Layer Awal**
   Kami memuat model VGG16 dengan parameter yang telah dilatih di ImageNet, mengecualikan fully connected layer (include_top=False), dan membekukan (freeze) seluruh lapisan pada awal pelatihan.
   
2. **Menambahkan Top Layers**
   Ditambahkan beberapa lapisan kustom (Flatten, Dense, Dropout) untuk melakukan klasifikasi berdasarkan kategori sampah (empat kelas).
   
3. **Compile Model untuk Training Awal**
   Model dikompilasi dengan learning rate yang relatif besar untuk proses training awal (feature extraction).
   
4. **Callbacks untuk Monitoring Training**
   Kami menggunakan callbacks seperti EarlyStopping dan ModelCheckpoint untuk mengontrol proses pelatihan dan mencegah overfitting.
   
5. **Training Awal (Feature Extraction)**
   Melatih model selama beberapa epoch untuk mengekstrak fitur secara awal.
   
6. **Fine-Tuning**
   Meng-unfreeze beberapa lapisan terakhir dari VGG16 untuk training lanjut dengan learning rate yang lebih kecil.
   
7. **Evaluasi Model**
   Model dievaluasi pada test set untuk mengukur akurasi klasifikasi setelah fine-tuning.

Berikut adalah kode lengkap yang digunakan, jalankan skrip pelatihan model:

```
# 1. Load VGG16 tanpa fully connected layer, freeze semua layer dulu
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

# 2. Tambah top layers
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# 3. Compile model dengan learning rate agak besar untuk training awal
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint("best_vgg16_model.h5", monitor='val_accuracy', save_best_only=True)
]

# 5. Training awal (feature extraction)
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=callbacks
)

# 6. Fine-tuning: unfreeze beberapa layer terakhir VGG16 untuk training lanjut
for layer in base_model.layers[-4:]:
    layer.trainable = True

# 7. Compile ulang dengan learning rate kecil untuk fine-tuning
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 8. Training fine-tuning
history_finetune = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    callbacks=callbacks
)

# 9. Evaluasi di test set
test_loss, test_acc = model.evaluate(test_generator)
print(f"Akurasi pada Test Set setelah Fine-Tuning: {test_acc * 100:.2f}%")
```

- **Menjalankan Notebook:**

Anda dapat membuka notebook pengembangan menggunakan Jupyter Notebook atau JupyterLab dengan perintah berikut:

[**Buka EcoSortAI Notebook di Google Colab**](https://colab.research.google.com/drive/1Oa2ke4XptuQCDGlJX0xpBgqiPqFkJePs#scrollTo=911Lw5um_DOE)

Pastikan untuk menjalankan seluruh sel notebook secara berurutan agar seluruh proses pengembangan mulai dari preprocessing, augmentasi, pelatihan, hingga evaluasi performa dapat direplikasi dengan benar.

---


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

- **Panduan Pengguna:**

Dokumen Pedoman Penggunaan Website EcoSortAI menyajikan petunjuk lengkap untuk membantu stakeholder, operator, dan masyarakat umum dalam mengakses serta memanfaatkan fitur-fitur website secara efektif. Untuk informasi lebih detail, silakan baca dokumen lengkap melalui tautan berikut:

[**Dokumen Pedoman Penggunaan Website EcoSortAI**](https://drive.google.com/file/d/1rJ_FtSmTP4w6pvLFkCzE8P8z9Y35YOfe/view?usp=sharing)

### Isi Dokumen

1. **Pendahuluan**  
   Selamat datang di pedoman penggunaan website EcoSortAI ([https://ecosortai.streamlit.app/](https://ecosortai.streamlit.app/)). Dokumen ini disusun untuk memberikan panduan mengenai cara mengakses dan memanfaatkan informasi serta analitik real-time terkait pengelolaan sampah berbasis teknologi AI.

2. **Akses Website**  
   - **Langkah 1:** Buka browser favorit Anda.  
   - **Langkah 2:** Masukkan URL berikut pada bilah alamat: [https://ecosortai.streamlit.app/](https://ecosortai.streamlit.app/).  
   - **Langkah 3:** Tunggu hingga halaman utama website termuat dengan tampilan antarmuka yang intuitif dan responsif.

3. **Navigasi Antarmuka**  
   Website EcoSortAI dirancang dengan antarmuka sederhana yang terdiri dari:  
   - **Header:** Menampilkan logo, judul, dan menu navigasi utama.  
   - **Dashboard Utama:** Menyajikan informasi real-time mengenai pengelolaan sampah, grafik, dan statistik hasil klasifikasi.  
   - **Panel Menu/Sidebar:** Menyediakan akses ke fitur pendukung seperti laporan historis, analisis data, dan informasi operasional.

4. **Fitur Utama**  
   a. **Pengambilan Data dan Klasifikasi Sampah**  
      - *Pengambilan Gambar:* Kamera pada aplikasi menangkap gambar sampah secara real-time.  
      - *Preprocessing:* Gambar disesuaikan ukurannya, dinormalisasi, dan (opsional) di-augmentasi untuk meningkatkan keragaman data.  
      - *Klasifikasi:* Gambar diklasifikasikan menggunakan model VGG16 dengan transfer learning dan fine-tuning ke dalam kategori "Anorganik Daur Ulang", "Anorganik Tidak Daur Ulang", "Organik", dan "B3".

   b. **Tampilan Dashboard Real-Time**  
      - Hasil klasifikasi ditampilkan secara langsung melalui dashboard interaktif.  
      - Pengguna dapat memantau status dan analitik pengelolaan sampah secara real-time guna mendukung pengambilan keputusan.

   c. **Pengelolaan Data dan Laporan**  
      - *Riwayat Data:* Seluruh data pengelolaan sampah tersimpan dan dapat diakses untuk analisis lanjutan.  
      - *Download Laporan:* Tersedia opsi untuk mengunduh laporan evaluasi performa sistem sebagai bahan evaluasi dan perencanaan.

5. **Video Demo**  
   Untuk memahami cara kerja EcoSortAI secara lebih mendalam, silakan tonton video demo berikut:  
   **Judul:** EcoSortAI: Real-time Waste Classification Demo  
   *(Tautan Video Demo dapat disesuaikan jika ada)*

6. **Tips Penggunaan**  
   - Pastikan koneksi internet stabil untuk tampilan data real-time yang optimal.  
   - Gunakan browser versi terbaru agar mendapatkan antarmuka yang responsif dan kinerja terbaik.  
   - Jika mengalami kendala, silakan merujuk kembali pada pedoman ini atau hubungi tim dukungan.

7. **Kontak dan Dukungan**  
   Untuk saran, pertanyaan, atau bantuan teknis terkait penggunaan website EcoSortAI, Anda dapat menghubungi kami melalui:
   - **Email:**  
     - A013XBF477@devacademy.id  
     - A296XBM496@devacademy.id  
     - A004XBM448@devacademy.id  
     - A382YBM063@devacademy.id

8. **Penutup**  
   Kami berharap dokumen pedoman ini dapat memudahkan Anda dalam menggunakan website EcoSortAI secara optimal. Terima kasih atas kepercayaan dan partisipasi Anda dalam mendukung pengelolaan sampah yang lebih inovatif dan berkelanjutan.

---
