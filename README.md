# Multimodal-ML-Housing-Price-Prediction-Using-Images-Tabular-Data
Predict housing prices using both structured data and house images.

# Multimodal Housing Price Prediction

## Project Overview

This project demonstrates a multimodal machine learning approach to predict housing prices by integrating both structured (tabular) data and visual information (images). The goal is to build a robust prediction model that leverages the complementary insights from different data types, potentially leading to more accurate price estimations than models relying on a single modality.

## Problem Statement

Accurate housing price prediction is crucial for real estate stakeholders. While tabular features (e.g., location, age, income) are commonly used, visual attributes (e.g., house condition, aesthetic appeal, surroundings) captured in images can provide invaluable additional context. This project aims to combine these modalities to enhance prediction accuracy.

## Objective

Develop and evaluate a multimodal regression model that:

1.  Extracts features from house images using Convolutional Neural Networks (CNNs).
2.  Integrates these image features with traditional tabular housing features.
3.  Trains a unified model capable of predicting housing prices.
4.  Evaluates model performance using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

## Dataset

**Tabular Data**: California Housing dataset, loaded from `california_housing_train.csv` and `california_housing_test.csv` available in the Colab `sample_data` directory. This dataset includes features such as `longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, and `median_income`, with `median_house_value` as the target variable.

**Image Data**: For demonstration purposes, and due to the absence of a paired image dataset for the California Housing data, **dummy image features** were generated. These dummy features simulate a scenario where each house has associated image data with an input shape of `(128, 128, 3)`.

## Methodology

The pipeline for multimodal housing price prediction involves several key steps:

1.  **Data Loading and Initial Preprocessing**: Both tabular (California Housing) and dummy image data were loaded. The tabular data was combined from training and test CSVs, and `median_house_value` was set as the target.

2.  **CNN Feature Extraction**: A simple CNN model (using `tensorflow.keras`) was constructed to process the image data. This CNN consisted of multiple `Conv2D` and `MaxPooling2D` layers followed by a `Flatten` layer to produce a feature vector for each image.

3.  **Tabular Data Preprocessing**: Numerical features in the tabular data were scaled using `StandardScaler` from `sklearn.preprocessing`. No categorical features were present in this specific dataset.

4.  **Feature Fusion**: The extracted image features and the preprocessed tabular features were concatenated horizontally (`np.concatenate` along `axis=1`) to create a unified `X_fused` dataset. The target variable `y_fused` was also prepared.

5.  **Data Splitting**: The `X_fused` and `y_fused` datasets were split into training and testing sets using `train_test_split` with an 80/20 ratio and a fixed `random_state` for reproducibility.

6.  **Model Architecture and Training**: A sequential Keras model was built for regression, consisting of several `Dense` layers with `relu` activation, and a final `Dense` layer with a single output for price prediction. The model was compiled with the `Adam` optimizer and `Mean Squared Error` (MSE) loss, tracking `Mean Absolute Error` (MAE).

7.  **Model Evaluation**: The trained model's performance was assessed on the unseen test set using MAE and RMSE.

## Results

After training for 20 epochs with a batch size of 32, the multimodal model achieved the following performance metrics on the test set:

*   **Mean Absolute Error (MAE): 51030.04**
*   **Root Mean Squared Error (RMSE): 71195.73**

These metrics indicate that, on average, the model's predictions deviated by approximately $51,030 from the actual housing prices.

## Usage

To run this project, open the Colab notebook. The cells are designed to be executed sequentially:

1.  **Install Dependencies**: Ensure TensorFlow and Scikit-learn are installed (Colab typically has these pre-installed, or they are installed by the notebook cells).
2.  **Load and Preprocess Data**: Execute the cells for loading tabular data and (dummy) image data.
3.  **Extract Features and Fuse**: Run the CNN feature extraction and tabular preprocessing cells, followed by the feature fusion cell.
4.  **Split Data**: Execute the data splitting cell.
5.  **Train Model**: Run the model definition and training cell.
6.  **Evaluate Model**: Execute the evaluation cell to see the performance metrics and training plots.

**Note**: To use real image data, you would need to replace the dummy image generation logic with your actual image loading and preprocessing steps, ensuring that images are correctly linked to their corresponding tabular entries.

## Dependencies

*   `pandas`
*   `numpy`
*   `tensorflow` (with `keras`)
*   `scikit-learn` (for `StandardScaler`, `ColumnTransformer`, `train_test_split`, `mean_absolute_error`, `mean_squared_error`)
*   `matplotlib` (for plotting training history)
