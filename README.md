# 🧠 Naïve Bayes Spam Comment Detector  

A Natural Language Processing (NLP) project that classifies YouTube comments as **spam** or **non-spam** using a **Multinomial Naïve Bayes** model.  
Developed by **Adilet Masalbekov**, Arcan Caglayan, and Muhammed Ikbal Ekinci for *COMP 237 – Introduction to Artificial Intelligence*.

---

## 🧩 Overview
- **Dataset:** [`Youtube01-Psy.csv`](https://archive.ics.uci.edu/dataset/380/youtube+spam+collection) from the *UCI Machine Learning Repository*  
- **Goal:** Automatically detect spam comments using text classification  
- **Techniques:** Count Vectorization + TF-IDF Transformation + Multinomial Naïve Bayes  
- **Evaluation:** Cross-validation and accuracy testing on held-out data  

> The script automatically downloads the dataset from UCI.  
> If the direct CSV is unavailable, it falls back to downloading and reading the official ZIP archive.

---

## 📊 Results
| Metric | Result |
|:--|:--|
| Cross-validation mean accuracy | **93.9 %** |
| Test set accuracy | **96.6 %** |
| Confusion matrix | 44 TP   41 TN   1 FP   2 FN |

The model reliably distinguishes spam from non-spam comments with minimal misclassifications.

---

## 🧠 Pipeline Summary
1. **Data Loading & Exploration** – Automatically downloads and loads the dataset from UCI.  
2. **Pre-processing & Vectorization** – Uses `CountVectorizer(stop_words='english')` to tokenize text.  
3. **TF-IDF Transformation** – Weights features by importance across comments.  
4. **Model Training** – Trains a `MultinomialNB` classifier on a stratified 75 / 25 split.  
5. **Cross-Validation & Testing** – Performs 5-fold CV and evaluates on the held-out test set.  
6. **Real-World Validation** – Classifies new user-written comments to demonstrate generalization.

---

## 🧰 Tech Stack
- **Python 3**
- **pandas**
- **scikit-learn**
- **requests**

---

## ⚙️ Run Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/adilet0212/naive-bayes-spam-detector.git
cd naive-bayes-spam-detector