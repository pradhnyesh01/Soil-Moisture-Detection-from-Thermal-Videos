# Soil-Moisture-Detection-from-Thermal-Videos
With the use of FLIR E75 Thermal footage, we aim to create a machine learning model that detects the presence of water in soil. 

To view and download the dataset, kindly follow the kaggle link:

https://www.kaggle.com/datasets/pradhnyeshkhorgade/thermal-footage-of-soil

### Project Overview
This project implements a machine learning pipeline to classify soil moisture conditions as dry or wet based on features extracted from thermal videos of soil. The approach extracts statistical temperature features (mean, standard deviation, max, min) from sampled video frames and trains a Random Forest classifier on aggregated video-level features. This tool can assist in agricultural monitoring, irrigation management, and environmental research by providing automated soil moisture detection from thermal imaging.

### Features
1. Efficient extraction of thermal statistical features from video frames.
2. Support for processing video datasets organized in dry/wet folders.
3. Saves extracted frames and detailed per-frame feature data for analysis.
4. Training and evaluation of a Random Forest classifier for binary classification.
5. Saves extracted frames and detailed per-frame feature data for analysis.
6. Predict soil moisture condition for new input videos using the trained model.

### Dependencies
1. Python 3.7+
2. OpenCV
3. NumPy
4. pandas
5. scikit-learn

Dependencies can be installed as per requirements.txt

### Results & Evaluation
The model achieves accuracy and detailed classification reports on the test split of the dataset, demonstrating reliable differentiation between dry and wet soil conditions based on thermal imaging.

### Future Work
1. Expand dataset to include different soil types and climatic conditions.
2. Integrate real-time video processing for field deployment.
3. Combine thermal imaging with other sensing modalities for robust estimation.

