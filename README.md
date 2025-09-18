AI & Machine Learning Project – CIFAR-10 Classification

This project explores different **machine learning and deep learning techniques** for image classification on the **CIFAR-10 dataset**.  
It compares traditional algorithms (Naive Bayes, Decision Trees) with deep learning models (MLP, VGG11 CNN) and feature extraction using **ResNet-18 + PCA**.

The goal is to evaluate how different approaches perform on the same dataset.


- **Python 3**
- **PyTorch** (deep learning, CNNs, MLPs, ResNet feature extraction)
- **scikit-learn** (Naive Bayes, Decision Tree, evaluation metrics)
- **NumPy / Torchvision** (data handling, CIFAR-10 dataset, transforms)


1. Clone this repository:
2. Install dependencies: pip install -r requirements.txt
3. Run the main pipeline: python main.py

Project Structure:
.
├── main.py                     # Orchestrates the full training & evaluation pipeline
├── data_loading.py             # CIFAR-10 loading, preprocessing & limiting per class
├── feature_extraction.py       # ResNet-18 feature extractor
├── dimensionality_reduction.py # PCA for dimensionality reduction
├── naive_bayes.py              # Custom & sklearn Naive Bayes
├── decision_tree.py            # Custom & sklearn Decision Tree
├── multi_layer_perceptron.py   # MLP model, training & evaluation
├── convolutional_neural_network.py # VGG11 CNN model
├── evaluation.py               # Accuracy, precision, recall, confusion matrix
├── dataset.json                # Example metadata
├── README.md                   # Project documentation
├── requirements.txt            # Dependencies

