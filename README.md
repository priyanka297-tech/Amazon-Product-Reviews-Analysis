# 💬 Sentiment Analysis Web App (Streamlit)

A professional and interactive **Sentiment Analysis Web Application** built using **Streamlit**, powered by a Machine Learning model (Logistic Regression) with **TF-IDF** and **SVD** for feature extraction and dimensionality reduction.

---

## 🚀 Features

* ✨ Clean and modern UI using Streamlit
* 🧠 Machine Learning-based sentiment prediction
* 📊 Supports **Negative**, **Neutral**, and **Positive** classification
* ⚡ Real-time text analysis
* 🎯 Optional confidence score display
* 🧩 Modular and easy-to-extend codebase

---

## 🛠️ Tech Stack

* **Python**
* **Streamlit**
* **Scikit-learn**
* **TF-IDF Vectorizer**
* **Truncated SVD**
* **Pickle (Model Serialization)**

---

## 📂 Project Structure

```
├── app.py                  # Main Streamlit application
├── sentiment_model.pkl     # Trained Logistic Regression model
├── tfidf.pkl               # TF-IDF vectorizer
├── svd.pkl                 # SVD transformer
└── README.md               # Project documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```
git clone https://github.com/your-username/sentiment-analysis-app.git
cd sentiment-analysis-app
```

### 2️⃣ Create Virtual Environment (Optional but Recommended)

```
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```

### 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

If you don’t have a `requirements.txt`, install manually:

```
pip install streamlit scikit-learn
```

---

## ▶️ Running the Application

```
streamlit run app.py
```

Then open your browser at:

```
http://localhost:8501
```

---

## 🧠 How It Works

1. User inputs a review in the text box
2. Text is **cleaned and preprocessed**
3. Converted into **TF-IDF features**
4. Reduced using **SVD (dimensionality reduction)**
5. Passed into the **Logistic Regression model**
6. Output sentiment is displayed with color-coded UI

---

## 📝 Sentiment Labels

| Label | Meaning     |
| ----- | ----------- |
| 0     | Negative 😡 |
| 1     | Neutral 😐  |
| 2     | Positive 😊 |

---

## 🎯 Example Inputs

* *"This product is amazing! I love it."* → Positive
* *"It's okay, not great but not bad."* → Neutral
* *"Worst experience ever. Totally disappointed."* → Negative

---

## 🙌 Acknowledgements

* Scikit-learn for ML tools
* Streamlit for rapid UI development



⭐ If you like this project, don’t forget to give it a star!
