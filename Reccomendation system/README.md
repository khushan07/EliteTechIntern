# 🎬 Movie Recommendation System

This project presents a **Movie Recommendation System** built using **Collaborative Filtering** techniques. It suggests movies to users based on their preferences and past interactions.

---

## 🔍 Project Objective

To build a recommender system that provides personalized movie suggestions using user-based and item-based collaborative filtering.

---

## 📊 Dataset

We use the **MovieLens 100K** dataset, which contains:

- `movies.csv`: Movie titles and genres  
- `ratings.csv`: User ratings for movies  

📌 Source: [GroupLens MovieLens 100K Dataset](https://grouplens.org/datasets/movielens/100k/)

---

## 🚀 Methodology

- Created a **user-item interaction matrix**
- Calculated **user-user similarity** and **item-item similarity** using cosine similarity
- Implemented **User-Based** and **Item-Based Collaborative Filtering**
- Generated top-N recommendations for users

---

## 📌 Features

- Personalized movie suggestions
- Easy-to-understand recommendations
- Interactive Jupyter Notebook
- Clean and modular structure

---

## 📁 Project Structure

movie-recommendation-system/
├── app_3.ipynb # Main notebook
├── movies.csv # Movie metadata
├── ratings.csv # User ratings
├── requirements.txt # Required libraries
├── README.md # Project overview
└── Outputs/ # (Optional) Generated outputs