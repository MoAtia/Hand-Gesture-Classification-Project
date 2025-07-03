# Hand Gesture Classification Project README

This README provides an overview of the steps implemented in the `Hand Gesture Classification Project.ipynb` Jupyter Notebook, which focuses on classifying hand gestures using machine learning models based on hand landmark data extracted from images or video.

## Project Overview
The notebook processes a dataset of hand landmarks (`hand_landmarks_data.csv`) extracted using MediaPipe, preprocesses the data, trains multiple machine learning models, and evaluates their performance for hand gesture classification. It also includes code for real-time gesture prediction using a webcam, though this section is commented out.

## Steps in the Notebook

### 1. Imports
The notebook begins by importing necessary Python libraries for data manipulation, visualization, preprocessing, and machine learning:
- **Data Handling**: `pandas`, `numpy`, `random`, `joblib`
- **Visualization**: `matplotlib.pyplot`, `seaborn`
- **Preprocessing**: `sklearn.preprocessing.StandardScaler`, `sklearn.preprocessing.LabelEncoder`
- **Modeling**: `sklearn.svm.SVC`, `sklearn.tree.DecisionTreeClassifier`, `sklearn.ensemble.RandomForestClassifier`, `xgboost.XGBClassifier`
- **Evaluation**: `sklearn.metrics.classification_report`, `sklearn.metrics.accuracy_score`, `sklearn.metrics.confusion_matrix`
- **Hand Landmark Processing**: `mediapipe`, `cv2` (OpenCV)

### 2. Make Code Reproducible
A function `set_seeds` is defined to set random seeds for `random` and `numpy` to ensure reproducibility of results. The seed is set to 42.

### 3. Load the Data
The dataset `hand_landmarks_data.csv` is loaded into a pandas DataFrame (`df`). The dataset contains 25,675 samples with 64 columns, including:
- 63 columns representing the x, y, z coordinates of 21 hand landmarks (labeled `x1` to `z21`).
- 1 column (`label`) indicating the gesture class (e.g., "call", "dislike", "fist", etc.).

The shape of the data is printed, and the first few rows are displayed using `df.head()`.

### 4. Data Exploration
#### 4.1 Visualize Sample
A function `plot_sample` is defined to visualize hand landmarks for a given sample:
- Extracts x and y coordinates for the 21 landmarks.
- Creates a scatter plot with an inverted y-axis to match image coordinates.
- Includes options (commented out) to connect landmarks with lines and display the gesture label.
- The plot is displayed with equal aspect ratio and a grid.

#### 4.2 Normalize Data
The data is normalized to improve model performance:
- **Wrist-based Normalization**: For each sample, the wrist landmark (point 0) is used as the origin, and coordinates are scaled by the distance between the wrist and the middle finger tip (point 12).
- The z-coordinates are dropped after normalization, as their mean and standard deviation are near zero, reducing the feature set to x and y coordinates only (42 features).

#### 4.3 Data Splitting
The normalized data is split into training and testing sets using `train_test_split`:
- **Features**: Normalized x and y coordinates.
- **Labels**: Gesture labels.
- **Split Ratio**: 80% training, 20% testing.
- **Random State**: 42 for reproducibility.

### 5. Modeling
Four machine learning models are trained and evaluated:

#### 5.1 Support Vector Machine (SVM)
- Model: `SVC` with `C=100.0` and `kernel='rbf'`.
- Trained on the training set, predictions are made on the test set.
- Performance is evaluated using `classification_report`, showing precision, recall, and F1-score for each gesture class.
- **Accuracy**: 98%.

#### 5.2 Decision Tree
- Model: `DecisionTreeClassifier` with default parameters.
- Trained and evaluated similarly to SVM.
- **Accuracy**: 94%.

#### 5.3 Random Forest
- Model: `RandomForestClassifier` with `n_estimators=150`, `min_samples_split=2`, and `max_depth=None`.
- Trained and evaluated similarly.
- **Accuracy**: 97%.

#### 5.4 XGBoost
- Labels are encoded using `LabelEncoder` to convert string labels to numerical values.
- Model: `XGBClassifier` with default parameters.
- Trained and evaluated, with predictions decoded back to original labels for reporting.
- **Accuracy**: 98%.

**Conclusion**: XGBoost is identified as the best-performing model based on accuracy and classification metrics.

### 6. Save Models
The trained models and label encoder are saved using `joblib` for future use:
- Label encoder: `chekpoints/label_encoder.pkl`
- XGBoost model: `chekpoints/xgb_model.pkl`
- Random Forest model: `chekpoints/rnf_model.pkl`

### 7. Real-Time Prediction (Commented Out)
A commented-out section demonstrates real-time hand gesture prediction using a webcam:
- **MediaPipe Setup**: Configures MediaPipe Hands for single-hand detection with a minimum confidence of 0.5.
- **Webcam Input**: Captures video frames, flips them horizontally, and converts to RGB.
- **Landmark Processing**: Extracts landmarks, normalizes them (same process as training data), and removes z-coordinates and wrist point.
- **Prediction**: Uses the trained Random Forest model to predict the gesture.
- **Visualization**: Draws landmarks on the frame and displays the predicted gesture label.
- **Exit Condition**: Pressing 'q' terminates the webcam feed.

## Requirements
To run the notebook, install the required libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost mediapipe opencv-python joblib
```

## Dataset
The dataset `hand_landmarks_data.csv` should be in the same directory as the notebook. It contains hand landmark coordinates extracted using MediaPipe, with gesture labels.

## Notes
- The real-time prediction code is commented out to avoid running webcam-based code unintentionally. Uncomment and run it in an environment with a webcam for testing.
- The normalization step is critical for consistent feature scaling across training and real-time prediction.
- The XGBoost model is recommended for deployment due to its superior performance.

## Usage
1. Ensure all dependencies are installed.
2. Place the dataset (`hand_landmarks_data.csv`) in the working directory.
3. Run the notebook cells sequentially to preprocess data, train models, and evaluate performance.
4. Optionally, uncomment the real-time prediction section to test with a webcam.