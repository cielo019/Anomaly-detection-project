# 🚀 Lightweight Anomaly Detection for IoT Network Traffic

##  Overview
This project presents a lightweight anomaly detection system designed for real-time network traffic monitoring in edge and IoT environments. The system leverages machine learning techniques to detect malicious activities such as botnets and DDoS attacks while maintaining low computational overhead.

---

##  Objective
- Detect anomalous network traffic in real time  
- Achieve high detection accuracy with low false positives  
- Optimize for edge/IoT deployment (low latency & resource usage)  

---

## Dataset
- **IoT-23 Dataset**
- Source: Zeek network flow logs (`conn.log.labeled`)
- Contains both **benign and malicious IoT traffic**

---

##  Methodology

###  Data Processing
- Removed Zeek metadata
- Replaced missing values (`- → 0`)
- Selected relevant network flow features
- Encoded categorical features (e.g., connection state)
- Normalized data using StandardScaler

---

### 🔹 Models Used

#### 1. Isolation Forest (Primary Model)
- Trained on **benign traffic only**
- Detects anomalies based on isolation principle

#### 2. Autoencoder (Comparison Model)
- Deep learning-based anomaly detection
- Uses reconstruction error for detection

---

## Model Optimization

###  Key Improvements
- **Benign-only training** for better anomaly detection
- **Contamination tuning** for balancing detection vs false positives
- **Feature reduction** for edge efficiency

---

## Results

### Final Optimized Model (Balanced Features)

| Metric | Value |
|------|------|
| Attack Recall | **87.7%** |
| Precision | **98.9%** |
| False Positive Rate | **~10%** |
| Latency per Flow | **~5.7 µs** |

---

###  Model Comparison

| Model | Features | Recall | FPR | Latency |
|------|--------|--------|------|--------|
| Full Model | 12 | 99.6% | ~10% | 8 µs |
| Reduced Model | 6 | 20% | ~10% | 5 µs |
| **Final Model** | **8** | **87.7%** | **~10%** | **5.7 µs** |

---

##  Key Insights
- Training on normal data improves anomaly detection
- Feature reduction must be balanced (too aggressive reduces performance)
- Port-based features are critical for detecting attacks
- Isolation Forest is highly suitable for edge deployment

---

##  Performance Highlights
- Real-time capable (microsecond latency)
- High precision and recall
- Lightweight and efficient
- Suitable for IoT/edge environments

---

##  Technologies Used
- Python
- Pandas
- Scikit-learn
- TensorFlow / Keras
- Google Colab

---

## Future Work
- Deploy on real IoT/edge devices
- Implement sliding window real-time detection
- Compare with additional models (One-Class SVM, LSTM)
- Multi-scenario dataset evaluation

---

##  Author
Hirushi Adikari (@cielo019)

---

##  References
- IoT-23 Dataset: https://www.stratosphereips.org/datasets-iot23  
- Isolation Forest Paper: Liu et al., IEEE ICDM 2008  

---
