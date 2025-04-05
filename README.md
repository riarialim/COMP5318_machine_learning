# COMP5318 Machine Learning and Data Mining
This unit offers a deep dive into the practical and theoretical aspects of machine learning and data mining, covering supervised learning, unsupervised learning, model selection, and evaluation techniques. 
Key topics include classification, feature engineering, ensemble methods, dimensionality reduction, and neural networks. Students apply these techniques on real-world datasets and gain hands-on experience with data pipelines and model building.

## Assignment 1: Image Classification using Fashion MNIST
### Summary
* Type: Individual project
* Objective: Develop classifiers from scratch to categorize grayscale clothing images into one of ten categories using the Fashion MNIST dataset.

### Problem
* Accurately classify 28x28 grayscale images of fashion items (e.g., shirts, shoes, bags) into 10 categories.

### Solution
* Implemented two classifiers—K-Nearest Neighbours (KNN) and Gaussian Naive Bayes (GNB)—in pure Python (without libraries like scikit-learn), incorporating:
  * Data preprocessing: Centering and dimensionality reduction via PCA and SVD.
  * Classifier development: Both KNN and GNB implemented from scratch.
  * Evaluation: Performance measured using accuracy, confusion matrices, and F1-scores, with KNN outperforming GNB in accuracy but at the cost of computational time.

### Tools & Technologies
* Python, NumPy, Matplotlib
* Techniques: PCA, SVD, KNN, Gaussian Naive Bayes
* Dataset: [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)
 

## Assignment 2: Image Classification using CIFAR-100
### Summary
* Type: Group project, 
* Collaborators: [Julio Correa](https://github.com/julio-correa-rios) and [Will Ryan](https://github.com/w-v-r)
* Objective: Evaluate multiple classifiers of increasing complexity on the CIFAR-100 dataset.

### Problem
* Classify 32x32 RGB images from the CIFAR-100 dataset into 20 coarse categories using three classifiers of varying complexity.

### Solution
* A comparative study was conducted using:
  * Gaussian Naive Bayes (GNB): A fast baseline with limited accuracy due to feature independence assumptions.
  * Random Forest: An ensemble method with improved accuracy and manageable computation time.
  * Convolutional Neural Network (CNN): Deep learning model offering the best performance, built with Keras using layers of convolution, pooling, dropout, and dense layers.

* Preprocessing steps included:
  * Standardisation
  * Wavelet Transform
  * ZCA Whitening
  Performance was evaluated using 10-fold cross-validation, confusion matrices, and accuracy metrics. CNN achieved ~48% accuracy, significantly outperforming others.

### Tools & Technologies
* Python, Scikit-learn, TensorFlow/Keras, NumPy, PyWavelets
* Techniques: Data preprocessing, Random Forest, CNN, Gaussian Naive Bayes
* Dataset: [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)