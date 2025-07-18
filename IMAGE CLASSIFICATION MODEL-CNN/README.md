# CIFAR-10 Image Classification using CNN

This project demonstrates how to build and evaluate a Convolutional Neural Network (CNN) model using TensorFlow and Keras for image classification on the **CIFAR-10** dataset. The model classifies images into 10 categories such as airplane, automobile, bird, cat, and more.

## 📁 Project Structure

```
image-classification-cnn/
├── cnn_cifar10_image_classifier.ipynb       # Main notebook for model training and evaluation
├── model.ipynb                              # Notebook to load and test saved model
├── cifar10_cnn_model.h5                     # Trained model file
├── outputs/                                 # Output visualizations
│   ├── confusion_matrix.png
│   ├── training_accuracy_plot.png
│   ├── training_loss_plot.png
│   ├── sample_predictions.png
│   └── misclassified_images.png
│   └── sample_output-2.png
├── README.md
├── requirements.txt
├── .gitignore
```

## 🚀 Model Summary

- **Dataset:** [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
- **Model:** Convolutional Neural Network (CNN)
- **Framework:** TensorFlow/Keras
- **Accuracy:** Achieved good accuracy on test set
- **Features:** 
  - Data normalization
  - Convolution + MaxPooling layers
  - Dense layers with Dropout
  - Visualization of training metrics
  - Confusion Matrix & Misclassified Images

## 🧪 Test the Model

Run `model.ipynb` to load the trained model (`cifar10_cnn_model.h5`) and make predictions on sample test images.

```python
model = load_model("cifar10_cnn_model.h5")
# Predict on one image from test set
predicted = model.predict(img[np.newaxis, ...])
```

## 📊 Output Visuals

Visualizations available under the `outputs/` folder:
- Accuracy & Loss plots
- Confusion Matrix
- Sample predictions
- Misclassified image examples

## 🔗 Dataset Source

The CIFAR-10 dataset can be downloaded directly via Keras:
```python
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
```

Alternatively, access the dataset info on:
👉 [Kaggle CIFAR-10 Dataset](https://www.kaggle.com/datasets/krishnaChaurasia/cifar-10)

## 📦 Requirements

Install dependencies using:
```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:
```
tensorflow
matplotlib
numpy
scikit-learn
```


## ## 🔍 Predict and Visualize a Single Image

You can test the model prediction on any individual test image using the following code:

```python
index = 100
img = x_test[index]
true_label = y_test[index]

img_input = np.expand_dims(img, axis=0)
pred_probs = model.predict(img_input)
predicted_label = np.argmax(pred_probs[0])

plt.imshow(img)
plt.title(f"True: {class_names[true_label[0]]} | Predicted: {class_names[predicted_label]}")
plt.axis('off')
plt.show()
```

### 🖼️ Output Example

This snippet will show the 101st image in the test set with its **true label** and **predicted label** displayed in the title. For example, it may display:

```
True: cat | Predicted: dog
```

And the image will be rendered using `matplotlib`.

---

## 💡 Inspiration

This project was developed as part of an internship task at **ELiteTech**, showcasing practical application of deep learning in computer vision.

---



