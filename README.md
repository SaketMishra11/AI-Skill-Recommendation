# AI-Skill-Recommendation

## 📌 Project Overview

This project is an **AI-based Career Recommendation System** that suggests the most suitable job roles based on the **skills entered by the user**.

The system uses **Machine Learning (Random Forest)** and **Natural Language Processing (TF-IDF)** to analyze the skills and predict the most relevant career paths.

The application is built with **Python and Streamlit**, which allows users to interact with the model through a **simple web interface**.

---

# 🎯 Problem Statement

Students and beginners often struggle to decide **which career path matches their skills**.

This project solves that problem by:

* Taking **user skills as input**
* Predicting **top career matches**
* Showing **required skills for that career**
* Showing **missing skills (skill gap analysis)**

---

# 🧠 Technologies Used

### Programming Language

* Python

### Libraries

* pandas
* numpy
* scikit-learn
* streamlit

### Machine Learning

* Random Forest Classifier

### NLP

* TF-IDF Vectorization

---

# 📂 Project Structure

```
career-recommendation-system
│
├── streamlit_app.py        # main application file
├── new_dataset.csv         # dataset containing job roles and skills
├── requirements.txt        # required libraries
└── README.md               # project documentation
```

---

# 📊 Dataset

The dataset contains two main columns:

| Column   | Description               |
| -------- | ------------------------- |
| skills   | skills required for a job |
| job_role | corresponding job role    |

Example:

| skills                         | job_role       |
| ------------------------------ | -------------- |
| python machine learning pandas | Data Scientist |
| python sql excel powerbi       | Data Analyst   |
| html css javascript react      | Web Developer  |

---

# ⚙️ How the System Works

The system follows this pipeline:

```
User Skills Input
        ↓
Text Preprocessing
        ↓
TF-IDF Vectorization
        ↓
Random Forest Model
        ↓
Top Career Predictions
        ↓
Required Skills
        ↓
Missing Skills
```

---

# 🔎 Step-by-Step Explanation

## 1️⃣ Loading the Dataset

The dataset is loaded using **pandas**.

```
df = pd.read_csv("new_dataset.csv")
```

This dataset contains job roles and their required skills.

---

## 2️⃣ Text Vectorization (TF-IDF)

Machine learning models cannot understand text directly.

So we convert the **skills text into numeric vectors** using TF-IDF.

```
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['skills'])
```

Example:

```
python machine learning
```

gets converted into numbers like:

```
[0.45, 0.32, 0.29]
```

---

## 3️⃣ Training the Machine Learning Model

We train a **Random Forest classifier**.

```
model = RandomForestClassifier()
model.fit(X, y)
```

Random Forest is used because:

* it works well for classification
* it reduces overfitting
* it handles complex patterns

---

## 4️⃣ Career Prediction

When a user enters skills:

```
python sql machine learning
```

the system:

1. converts skills into TF-IDF vector
2. sends it to the trained model
3. predicts the most suitable job roles

---

## 5️⃣ Top Career Recommendations

Instead of showing only one job, the system uses:

```
predict_proba()
```

to find **top matching careers**.

Example output:

```
Data Scientist → 70%
Data Analyst → 30%
```

---

## 6️⃣ Required Skills

Once the best job is predicted, the system retrieves **skills required for that job** from the dataset.

Example:

```
Data Scientist
```

Required Skills:

```
python
machine learning
statistics
pandas
```

---

## 7️⃣ Skill Gap Analysis

The system compares:

```
User Skills
vs
Required Skills
```

and identifies **missing skills**.

Example:

User Skills:

```
python sql
```

Missing Skills:

```
machine learning
statistics
pandas
```

---

# 🌐 Streamlit Web Interface

Streamlit is used to build the web application.

Features of the interface:

* user can enter skills
* shows top career matches
* displays required skills
* highlights missing skills

Example UI:

```
🚀 AI Career Recommendation System

Enter your skills:
python sql machine learning

Top Career Matches

1 Data Scientist → 70%
2 Data Analyst → 30%

Required Skills
Missing Skills
```

---

# ▶️ How to Run the Project

## 1️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

## 2️⃣ Run the application

```
streamlit run streamlit_app.py
```

---

## 3️⃣ Open in browser

```
http://localhost:8501
```

---

# 🚀 Deployment

The project can be deployed using:

* **Streamlit Cloud**
* GitHub integration

After deployment the app becomes accessible online.

Example:

```
https://your-app-name.streamlit.app
```

---

# 💡 Future Improvements

Possible upgrades:

* resume upload and analysis
* skill learning recommendations
* job market demand analysis
* integration with LinkedIn data

---

# 👨‍💻 Author

**Saket Mishra**

B.Tech Artificial Intelligence & Data Science Student
Passionate about Machine Learning and Data Science.

---
