
# Human Activity Recognition Using Smartphone Sensor Data

## 1. Project Overview

This project focuses on Human Activity Recognition (HAR) using smartphone sensor data.

The goal is to classify human activities into six different classes based on sensor measurements collected from smartphones.

### Activities

1. WALKING
2. WALKING_UPSTAIRS
3. WALKING_DOWNSTAIRS
4. SITTING
5. STANDING
6. LAYING

The project uses the UCI Human Activity Recognition Using Smartphones Dataset.

---

## 2. Dataset

The dataset contains:

- 10,299 total samples
- 561 sensor features
- 6 activity classes
- 7,352 training samples
- 2,947 test samples

The raw dataset is not included in this repository.

---

## 3. Project Workflow

```text
Raw Smartphone Sensor Data
        ↓
Data Loading
        ↓
Data Inspection
        ↓
Data Preprocessing
        ↓
Train / Validation Split
        ↓
Feature Standardization
        ↓
Baseline Models
        ↓
Our MLP Model
        ↓
Model Evaluation
        ↓
Model Comparison
        ↓
Final Analysis
