# 🚀 Lightweight Anomaly Detection for IoT & Network Traffic

## 📌 Overview
This project presents a lightweight anomaly detection system designed for real-time network traffic monitoring in edge and IoT environments. The system leverages machine learning techniques to detect malicious activities such as botnets and DDoS attacks while maintaining low computational overhead and microsecond-level latency.

---

## 🎯 Objective
- Detect anomalous network traffic in real time  
- Achieve high detection accuracy with low false positives  
- Optimize for edge/IoT deployment (low latency & resource usage)  
- Validate performance across multiple datasets  

---

## 📊 Dataset

### 🔹 IoT-23 Dataset
- Source: Zeek network flow logs (`conn.log.labeled`)
- Contains both **benign and malicious IoT traffic**

### 🔹 CICIDS2017 Dataset
- Real-world network traffic dataset
- Includes diverse attack types (e.g., DDoS)
- Used to evaluate **model generalization**

---

## ⚙️ Methodology

### 🔹 Data Processing
- Removed metadata  
- Replaced missing and infinite values  
- Selected lightweight network flow features  
- Converted labels (Benign = 0, Attack = 1)  
- Normalized data using StandardScaler  

---

### 🔹 Models Used

#### 1. Isolation Forest (Primary Model)
- Trained on **benign traffic only**
- Detects anomalies based on isolation principle
- Lightweight and suitable for real-time detection

#### 2. Autoencoder (Comparison Model)
- Deep learning-based anomaly detection
- Uses reconstruction error for anomaly detection

#### 3. Supervised Models (Benchmarking)
- Random Forest  
- Support Vector Machine (SVM)  
- Used for performance comparison

---

## ⚡ Model Optimization

### 🔹 Key Improvements
- **Benign-only training** for better anomaly detection  
- **Contamination tuning** for balancing detection vs false positives  
- **Feature reduction** for edge efficiency  
- **Threshold tuning** for improved detection  

---

## 📈 Results

### 🔹 IoT-23 Dataset (Final Model)

| Metric | Value |
|------|------|
| Attack Recall | **87.7%** |
| Precision | **98.9%** |
| False Positive Rate | **~10%** |
| AUC Score | **0.93** |
| Latency per Flow | **~5 µs** |

---

### 🔹 CICIDS2017 Dataset

| Metric | Value |
|------|------|
| Attack Recall | **63.4%** |
| Precision | **89.2%** |
| False Positive Rate | **~10%** |
| AUC Score | **0.81** |
| Latency per Flow | **~3.7 µs** |

---

### 🔹 Cross-Dataset Comparison

| Dataset | Recall | Precision | FPR | AUC |
|--------|--------|----------|------|------|
| IoT-23 | 87.7% | 98.9% | ~10% | 0.93 |
| CICIDS2017 | 63.4% | 89.2% | ~10% | 0.81 |

---

### 🔹 Model Comparison

| Model | Recall | FPR | Training Time | Type |
|------|--------|------|---------------|------|
| Isolation Forest | 87.7% | ~10% | Low | Unsupervised |
| Random Forest | 100% | 0% | Medium | Supervised |
| SVM | 99.2% | 0.3% | High | Supervised |

---

## 🧠 Key Insights
- Training on normal data significantly improves anomaly detection  
- Feature reduction must be balanced (too aggressive reduces performance)  
- Isolation Forest performs better than deep learning for tabular data  
- Performance decreases on complex datasets, showing realistic behavior  
- Lightweight models provide the best trade-off between accuracy and efficiency  

---

## ⚠️ Limitations
- Reduced recall on complex datasets (CICIDS2017)  
- Fixed thresholds may not adapt to dynamic environments  
- Difficulty capturing highly diverse attack patterns  

---

## 🚀 Performance Highlights
- Real-time capable (microsecond latency)  
- High precision and low false alarms  
- Lightweight and efficient  
- Suitable for IoT/edge environments  

---

## 🛠 Technologies Used
- Python  
- Pandas  
- Scikit-learn  
- TensorFlow / Keras  
- Google Colab  

---

## 🔮 Future Work
- Deploy on real IoT/edge devices  
- Adaptive threshold tuning  
- Hybrid ML + Deep Learning models  
- Real-time streaming detection  
- Evaluation on additional datasets  

---

## 👩‍💻 Author
Hirushi Adikari (@cielo019)

---

## 📚 References
- IoT-23 Dataset: https://www.stratosphereips.org/datasets-iot23  
- CICIDS2017 Dataset: https://www.unb.ca/cic/datasets/ids-2017.html  
- Isolation Forest Paper: Liu et al., IEEE ICDM 2008  

---
