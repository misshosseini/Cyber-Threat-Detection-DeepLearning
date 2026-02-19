# 🔐 Deep Learning-Based Cyber Threat Detection

This project implements a deep learning model to detect suspicious cyber activities using the BETH (Behavioral Event Threat Hunting) dataset.

## 📊 Dataset

The BETH dataset simulates real-world system logs and contains process-level behavioral features such as:

- processId
- threadId
- parentProcessId
- userId
- mountNamespace
- argsNum
- returnValue
- sus_label (Target Variable)

The target label `sus_label` indicates:
- 0 → Benign Event
- 1 → Suspicious Event

## 🧠 Model Architecture

A Fully Connected Neural Network (MLP) implemented using PyTorch:

- Input Layer (7 features)
- Hidden Layer (128 neurons, ReLU)
- Hidden Layer (64 neurons, ReLU)
- Output Layer (1 neuron, Sigmoid)

## ⚙️ Training Setup

- Loss Function: Binary Cross Entropy
- Optimizer: SGD (with weight decay)
- Feature Scaling: StandardScaler
- Evaluation: Accuracy on Train, Validation, and Test sets

## 📈 Results

The model was evaluated using:

- Training Accuracy
- Validation Accuracy
- Test Accuracy

(You can add your actual numbers here)

## 🎯 Objective

The goal of this project is to demonstrate how deep learning models can detect anomalous process behavior and enhance cybersecurity defenses.

## 🚀 Future Improvements

- Use BCEWithLogitsLoss for better numerical stability
- Add Precision, Recall, and F1-score evaluation
- Handle class imbalance
- Deploy as a REST API for real-time threat detection
