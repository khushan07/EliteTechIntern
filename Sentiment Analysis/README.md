# 📊 Sentiment Analysis with NLP | ELiteTech Internship

This notebook performs **Sentiment Analysis** on customer reviews using **TF-IDF Vectorization** and **Logistic Regression**.

---

## 🔍 Project Objective

The goal is to build a simple and effective NLP pipeline to classify reviews as **Positive** or **Negative**. This is part of the **ELiteTech Internship Program**.

---

## 📁 Dataset Description

- `Text`: Customer review
- `Sentiment`: 
  - `1` = Positive  
  - `0` = Negative

  ## Dataset

This project uses the **Amazon Fine Food Reviews** dataset for sentiment analysis.

Due to GitHub's 25 MB file size limit, the full dataset (`Reviews.csv`, ~246 MB) is **not included** in the repository.

👉 [Download the full dataset from Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

After downloading, place the `Reviews.csv` file in the project directory to run the complete model training.



---

## ⚙️ Tech Stack

- Python (Jupyter Notebook)
- Pandas, NumPy
- scikit-learn
- Matplotlib, Seaborn
- Regular Expressions (re)
- NLTK

---

## ✅ Workflow Summary

1. **Import Libraries**
2. **Load Dataset**
3. **Preprocess Text**
   - Lowercase
   - Remove HTML tags, punctuations
   - Remove stopwords, lemmatization
4. **TF-IDF Vectorization**
5. **Train Logistic Regression Model**
6. **Evaluate with Accuracy, Confusion Matrix, Classification Report**
7. **Real-time Prediction Function**
8. **Test on Sample Reviews**

---


## 🧠 Real-Time Prediction Example

```python
sample_reviews = ["This taffy is so good.  It is very soft and chewy.  The flavors are amazing.  I would definitely recommend you buying it.  Very satisfying!",
 "Product labeled as jumbo salted peanuts but contained small, unsalted ones. Unclear if it was a mistake or mislabeling.."]
for i, review in enumerate(sample_reviews,1):

    print(f"\n Sample Review {i}: \n{review}")
    print("Predicted Sentiment:", predict_sentiment(review))

```


```python
Review: This taffy is so good. It is very soft and chewy...
Prediction: Positive :-)

Review: Product labeled as jumbo salted peanuts but...
Prediction: Negative :-(
```
