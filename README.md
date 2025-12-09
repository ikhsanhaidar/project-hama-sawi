# 🐛 PestGuard - AI Pest Detection System

**PestGuard** adalah sistem deteksi hama otomatis berbasis _Deep Learning_ yang menggunakan 4 model CNN berbeda untuk mendeteksi keberadaan hama pada tanaman sawi. Sistem ini dilengkapi dengan dashboard admin untuk monitoring performa model.

![PestGuard Banner](https://img.shields.io/badge/PestGuard-AI%20Pest%20Detection-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange)
![Flask](https://img.shields.io/badge/Flask-2.3-lightgrey)

## 🚀 **Fitur Utama**

### 📊 **Dashboard Admin**

- Visualisasi performa 4 model CNN
- Grafik akurasi dan loss training
- Matriks kebingungan (confusion matrix)
- Perbandingan statistik model
- Rekomendasi model terbaik

### 🖼️ **Deteksi Hama**

- Analisis gambar dengan 4 model:
  1. **CNN From Scratch** - Akurasi: 96%
  2. **VGG16 Transfer Learning** - Akurasi: 95%
  3. **Xception Model** - Akurasi: 95%
  4. **NASNetMobile** - Akurasi: 96%
- Support format: JPG, PNG, JPEG, GIF
- Preprocessing otomatis (resize, normalisasi RGB)

### 📈 **Analisis Model**

- Metrics detail: Precision, Recall, F1-Score
- Visualisasi training proses
- Perbandingan performa antar model
- Export laporan otomatis

## 🛠️ **Instalasi & Setup**

### **Prerequisites**

- Python 3.8 atau lebih tinggi
- pip (Python package manager)
- Virtual environment (disarankan)

### **1. Clone Repository**

```bash
git clone https://github.com/username/PestGuard.git
cd PestGuard
```
