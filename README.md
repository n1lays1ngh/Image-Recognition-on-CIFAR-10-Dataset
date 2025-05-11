# Image Classification on CIFAR-10 Using Convolutional Neural Networks (CNN)

This project demonstrates image classification using a Convolutional Neural Network (CNN) implemented in Python with TensorFlow and Keras. It utilizes the popular CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 different classes.

## 📌 Overview

The objective of this project is to build a CNN model capable of accurately classifying images from the CIFAR-10 dataset. The notebook includes data preprocessing, model architecture design, training, evaluation, and visualization of results.

## 📂 Dataset

- **Dataset:** [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
- **Description:** The dataset consists of 60,000 images divided into 10 classes with 6,000 images per class.
- **Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

The dataset is automatically downloaded using Keras' built-in datasets module.

## 🧠 Model Architecture

The CNN model consists of:

- 3 Convolutional layers with ReLU activation
- MaxPooling layers for downsampling
- Dropout layers for regularization
- Fully connected Dense layers
- Softmax output layer for multiclass classification

**Loss Function:** Categorical Crossentropy  
**Optimizer:** Adam  
**Evaluation Metric:** Accuracy

## 📈 Results

- **Training Accuracy:** ~99%
- **Test Accuracy:** ~70%
- **Training Epochs:** 50  
- Model shows strong performance on the training set with moderate generalization to unseen data.

### 📊 Visualizations

The notebook includes:

- Accuracy and loss curves
- Sample predictions with actual vs. predicted labels
- Confusion matrix for detailed performance analysis

---

## ⚙️ Installation

### 🔧 Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/cifar10-cnn-classifier.git
   cd cifar10-cnn-classifier
   ```

2. **Create a virtual environment (optional but recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook IMAGE_RECOGNITION_ON_CIFAR_10_DATASET_USING_CONVOLUTIONAL_NEURAL_NETWORK.ipynb
   ```

---

## 📦 Dependencies

See `requirements.txt` for full package list. Key dependencies:

- `tensorflow`
- `keras`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `jupyter`

---

## 🤝 Contributing

Feel free to fork the repo, raise issues, or submit pull requests. Contributions are always welcome!

---

## 🪪 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- TensorFlow & Keras Documentation
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- Community tutorials and research


