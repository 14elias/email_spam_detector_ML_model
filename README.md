# 📧 ML-Based Spam Email Detector

A **Machine Learning** project that detects spam emails using Natural Language Processing (NLP).  
The model is trained on the **Enron Spam Dataset** using **TF-IDF vectorization** and **Logistic Regression** for high-accuracy classification.

The application includes a **Streamlit web interface** for real-time predictions and supports **FastAPI** for API-based deployment.

---

## 🛠️ Tech Stack

- Python
- Pandas – Data preprocessing
- scikit-learn – Machine learning & pipeline
- TF-IDF Vectorizer – Text feature extraction
- Logistic Regression – Classification model
- Matplotlib & Seaborn – Data visualization
- Streamlit – Web interface
- FastAPI – Backend API framework
- Joblib – Model serialization

---

## 📊 Dataset

- Enron Spam Email Dataset
- 33,000+ email messages
- Binary classification:
  - `0` → Ham (Not Spam)
  - `1` → Spam

---

## ⚙️ Project Workflow

### 1️⃣ Data Preprocessing

- Removed missing email messages
- Converted labels (`ham → 0`, `spam → 1`)
- Train-test split (80% training, 20% testing)

### 2️⃣ Feature Engineering

- TF-IDF Vectorizer:
  - English stop words removal
  - N-grams (1,2)
  - `max_df = 0.9`
  - `min_df = 3`

### 3️⃣ Model Training

- Logistic Regression (`liblinear`, `max_iter=1000`)
- Implemented using `sklearn.pipeline.Pipeline`

### 4️⃣ Model Evaluation

- Accuracy: **99%**
- Precision, Recall, F1-score ≈ **0.99**
- Confusion matrix saved as:
  ```
  confusion_matrix.png
  ```

### 5️⃣ Model Saving

- Trained model saved as:
  ```
  spam_classifier.pkl
  ```

---

## 🚀 How to Run the Project

### 🔹 1. Install Dependencies

```bash
pip install pandas scikit-learn matplotlib seaborn streamlit fastapi joblib
```

---

### 🔹 2. Train the Model

```bash
python train.py
```

This generates:

- `spam_classifier.pkl`
- `confusion_matrix.png`

---

### 🔹 3. Run the Streamlit App

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal.

---

## 🌐 Streamlit App Features

- Text input for email content
- One-click evaluation
- Instant prediction:
  - ✅ Not Spam
  - 🚫 Spam

---

## 🔌 (Optional) FastAPI Endpoint

Example API route:

```python
@app.post("/")
def index(text: str):
    result = model.predict([text])[0]
    return "Spam" if result == 1 else "Not Spam"
```

Run the API server:

```bash
uvicorn app:app --reload
```

---

## 📈 Model Performance

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 99%   |
| Precision | 0.99  |
| Recall    | 0.99  |
| F1-Score  | 0.99  |

---

## 📂 Project Structure

```
├── enron_spam_data.csv
├── train.py
├── app.py
├── spam_classifier.pkl
├── confusion_matrix.png
└── README.md
```

---

## 💡 Future Improvements

- Cloud deployment (AWS / Render / Railway)
- Show prediction probability
- Improved UI design
- Docker support
- Unit testing

---

⭐ If you like this project, give it a star!
