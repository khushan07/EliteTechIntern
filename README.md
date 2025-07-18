# 🧠 Internship Projects – ELiteTech Pvt. Ltd.

This repository contains the tasks I completed during my internship at **ELiteTech Pvt Ltd**.  
These projects gave me **hands-on experience with Machine Learning, Deep Learning, and Natural Language Processing** techniques. Below is a list of the key projects I worked on.

---

## 📌 1. Decision Tree Classification

Implemented a **Decision Tree** model using **scikit-learn** to classify whether a movie would be a **Hit** or **Flop** based on pre-release features like budget, popularity, and vote count.

**Highlights**:
- Used **Entropy** as splitting criteria.
- Visualized the decision path using `plot_tree`.
- Evaluated performance using **accuracy**, **confusion matrix**, and **classification report**.

📁 Location: `decision-tree-movie-classification/`

---

## 💬 2. Sentiment Analysis

Built an **NLP pipeline** to classify customer review texts into **Positive** or **Negative** sentiments.

**Highlights**:
- Text preprocessing: tokenization, stopword removal, lemmatization.
- Feature extraction using **TF-IDF Vectorizer**.
- Trained classifiers using **Logistic Regression** and evaluated performance.

📁 Location: `Sentiment-Analysis/`

---

## 🖼️ 3. Image Classification 

Developed an **image classification** model using **Convolutional Neural Networks (CNNs)** to classify images into categories.

**Highlights**:
- Built using **Keras** with a **TensorFlow** backend.
- Applied **data augmentation** to enhance generalization.
- Trained and evaluated on datasets like **MNIST** or **CIFAR-10**.

📁 Location: `IMAGE CLASSIFICATION MODEL-CNN/` 

---

## 🎬 4. Recommendation System

Created a **movie recommendation system** using **Collaborative Filtering** and **Genre-Based Hybrid Filtering** on the [MovieLens 100K Dataset](https://grouplens.org/datasets/movielens/100k/).

**Highlights**:
- Built a **user-item interaction matrix**.
- Used **Cosine Similarity** for user-based and item-based recommendations.
- Evaluated using metrics like **precision@k** and **rating distribution analysis**.

📁 Location: `Reccomendation system/`

---

## 🛠️ Tech Stack

- **Languages**: Python  
- **Libraries**:
  - `NumPy`, `Pandas`
  - `Scikit-learn`, `Matplotlib`, `Seaborn`
  - `NLTK`, `spaCy`
  - `TensorFlow`, `Keras`
- **Tools**: Jupyter Notebook, VS Code

---

## 📂 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/khushan07/EliteTechIntern.git
   cd EliteTechIntern
   ```
2. Go to any project folder (e.g., recommendation-system/) and install dependencies:
  ```bash

  pip install -r requirements.txt
  ```
3. Launch the Jupyter notebook:
```bash
  jupyter notebook
```
