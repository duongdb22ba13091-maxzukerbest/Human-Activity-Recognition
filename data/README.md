
# Dataset

This project uses the **UCI Human Activity Recognition Using Smartphones Dataset**.

## Dataset Information

The dataset contains smartphone sensor measurements collected from participants performing six activities:

1. WALKING
2. WALKING_UPSTAIRS
3. WALKING_DOWNSTAIRS
4. SITTING
5. STANDING
6. LAYING

Dataset size:

- Total samples: 10,299
- Training samples: 7,352
- Test samples: 2,947
- Features: 561
- Classes: 6

## Download

The original dataset can be downloaded from the UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

## Repository Policy

The raw dataset is not included in this GitHub repository.

Users should download the original dataset from UCI and place it in the appropriate `data/` directory before running the notebooks.

## Dataset Structure

After downloading and extracting the dataset, the structure should contain:

```text
UCI HAR Dataset/
├── activity_labels.txt
├── features.txt
├── features_info.txt
├── README.txt
├── train/
└── test/
