# 🧠 Naïve Bayes Spam Comment Detector  

A Natural Language Processing (NLP) project that classifies YouTube comments as **spam** or **non-spam** using a **Multinomial Naïve Bayes** model.  
Developed by **Adilet Masalbekov**, Arcan Caglayan, and Muhammed Ikbal Ekinci for *COMP 237 – Introudction to Artificial Intelligence*.

---

## 🧩 Overview
- **Dataset:** [`Youtube01-Psy.csv`](https://archive.ics.uci.edu/dataset/380/youtube+spam+collection) from the *UCI Machine Learning Repository*  
- **Goal:** Automatically detect spam comments using text classification  
- **Techniques:** Count Vectorization + TF-IDF Transformation + Multinomial Naïve Bayes  
- **Evaluation:** Cross-validation and accuracy testing on held-out data  

---

## 📊 Results
| Metric | Result |
|:--|:--|
| Cross-validation mean accuracy | **93.87 %** |
| Test set accuracy | **95.45 %** |
| Confusion matrix | 42 TP  |  42 TN  |  3 FP  |  1 FN |

The model reliably distinguishes spam from non-spam comments with minimal misclassifications :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}.

---

## 🧠 Pipeline Summary
1. **Data Loading & Exploration** – Inspected dataset for structure and balance.  
2. **Pre-processing & Vectorization** – Used `CountVectorizer` with English stop words to tokenize text.  
3. **TF-IDF Transformation** – Weighted features by importance across comments.  
4. **Model Training** – Trained `MultinomialNB` on 75 % of the data.  
5. **Cross-Validation & Testing** – Performed 5-fold CV and evaluated on the remaining 25 %.  
6. **Real-World Validation** – Successfully classified new manually written comments.  

---

## 🧰 Tech Stack
- **Python 3**
- **Pandas**
- **scikit-learn**
- **NLTK**

---

## ⚙️ Run Locally
Clone the repository and run the Python script:

```bash
git clone https://github.com/adilet0212/naive-bayes-spam-detector.git
cd naive-bayes-spam-detector
python project_script.py
