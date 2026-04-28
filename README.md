# Machine Learning Projects

A collection of end-to-end machine learning and data science projects covering classification, regression, recommendation systems, natural language processing, and computer vision.

---

## Table of Contents

1. [Credit Card Fraud Detection](#1-credit-card-fraud-detection)
2. [House Price Prediction](#2-house-price-prediction)
3. [Human Activity Recognition](#3-human-activity-recognition)
4. [Movie Recommender System](#4-movie-recommender-system)
5. [Supermarket Sales Analysis](#5-supermarket-sales-analysis)
6. [ChatBot](#6-chatbot)
7. [Vehicle Number Plate Detection](#7-vehicle-number-plate-detection)
8. [Tech Stack](#tech-stack)

---

## 1. Credit Card Fraud Detection

**Directory:** `Credit Card Fraud Detection System/`

### Objective
Detect fraudulent credit card transactions from a highly imbalanced real-world dataset.

### Dataset
- `creditcard.csv` — 285,962 transactions (99.83% legitimate, 0.17% fraudulent)

### Approach
- Separated legitimate and fraudulent transactions
- Applied **under-sampling** to balance classes (492 samples each)
- 80/20 stratified train-test split

### Model
| Algorithm | Training Accuracy | Test Accuracy |
|---|---|---|
| Logistic Regression | ~95% | ~95% |

### Key Concepts
- Imbalanced dataset handling
- Under-sampling strategy
- Binary classification

---

## 2. House Price Prediction

**Directory:** `House Price Prediction/`

### Objective
Predict house prices based on property features such as area, number of rooms, and amenities.

### Dataset
- `Housing.csv` — Real estate property records

### Features
Area, Bedrooms, Bathrooms, Stories, Parking, Air Conditioning, Hot Water Heating, Main Road access, Guest Room, Basement, Preferred Area, Furnishing Status

### Approach
- Exploratory Data Analysis (scatter plots, box plots, count plots)
- Label Encoding for categorical variables
- Feature scaling with `StandardScaler`
- Multi-model comparison

### Models Compared
| Model | Metric |
|---|---|
| Linear Regression | MSE, R², MAE |
| Random Forest Regressor | MSE, R², MAE |
| Gradient Boosting Regressor | MSE, R², MAE |
| Ridge Regression | MSE, R², MAE |
| Support Vector Regressor | MSE, R², MAE |

---

## 3. Human Activity Recognition

**Directory:** `Human activity recoganation/`

### Objective
Classify six types of human physical activities using smartphone sensor data.

### Dataset
- UCI HAR Dataset — 561 accelerometer and gyroscope features
- **Training:** 7,352 samples | **Test:** 2,947 samples

### Activities
| Label | Activity |
|---|---|
| 0 | Laying |
| 1 | Sitting |
| 2 | Standing |
| 3 | Walking |
| 4 | Walking Downstairs |
| 5 | Walking Upstairs |

### Approach
- Visualized activity distribution across subjects
- Built and compared multiple Deep Neural Network configurations

### Model Architecture
- Embedding → LSTM → Dropout → Dense (Softmax)
- Optimizers tested: SGD, RMSprop, **Adam** (best performance)
- Trained for 50 epochs

---

## 4. Movie Recommender System

**Directory:** `Movie Recommandation System/`

### Objective
Recommend movies similar to a given title using content-based filtering.

### Dataset
- `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` — TMDB 5000 movie dataset
- 4,813 movies after preprocessing

### Approach
1. Merged movies and credits datasets on title
2. Extracted genres, keywords, top 3 cast members, and director
3. Combined all features into unified "tags" per movie
4. Vectorized tags and computed **Cosine Similarity** matrix
5. Returns top 5 most similar movies for any input title

### Output
Serialized artifacts (`movie_list.pkl`, `similarity.pkl`, `movie_dict.pkl`) ready for deployment in a web app (e.g., Streamlit).

---

## 5. Supermarket Sales Analysis

**Directory:** `Supermarket Analysis/`

### Objective
Analyze supermarket sales patterns, customer behavior, and build classification models on sales data.

### Dataset
- `supermarket_sales - Sheet1.csv` — Transactions across multiple branches

### EDA Highlights
- Product line distribution and revenue breakdown
- Gender-based purchasing patterns
- Branch-wise and payment method analysis
- Sales trends over time
- Average ratings by product line

### Models Compared
| Model | Metrics |
|---|---|
| Logistic Regression | Accuracy, F1, Confusion Matrix |
| Decision Tree | Accuracy, F1, Confusion Matrix |
| Random Forest | Accuracy, F1, Confusion Matrix |
| Gradient Boosting | Accuracy, F1, Confusion Matrix |
| XGBoost | Accuracy, F1, Confusion Matrix |
| Extra Trees | Accuracy, F1, Confusion Matrix |

---

## 6. ChatBot

**Directory:** `ChatBot/`

### Objective
Build an intent-based conversational chatbot using NLP and deep learning.

### Dataset
- `contents.json` — Predefined intents with patterns and responses

### Pipeline
1. Load JSON intent data
2. Tokenize and pad input sequences (vocabulary: 2,000 words)
3. Label encode intent tags

### Model Architecture
```
Input → Embedding (vocab+1, dim=10) → LSTM (10 units) → Flatten → Dense (Softmax)
```
- **Optimizer:** Adam
- **Loss:** Sparse Categorical Crossentropy
- **Epochs:** 300

### Output
An interactive chat loop that maps user input to predicted intents and returns a random matching response.

---

## 7. Vehicle Number Plate Detection

**Directory:** `Vechile Number Plate/`

### Objective
Detect and extract text from vehicle number plates in images using computer vision and OCR.

### Tech
- **OpenCV** — Image preprocessing and contour detection
- **Pytesseract** — OCR for character recognition

### Pipeline
1. Apply Gaussian Blur and convert to grayscale
2. Sobel edge detection followed by OTSU thresholding
3. Morphological CLOSE operation to merge characters
4. Contour detection with aspect ratio and white pixel validation
5. Extract candidate plate region and run Pytesseract OCR

### Output
Extracted license plate text printed for each detected plate region in input images.

---

## Tech Stack

| Category | Libraries / Tools |
|---|---|
| Data Manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Machine Learning | `scikit-learn`, `xgboost` |
| Deep Learning | `tensorflow`, `keras` |
| NLP | `nltk`, `keras Tokenizer` |
| Computer Vision | `opencv-python`, `pytesseract` |
| Recommendation | Cosine Similarity (`sklearn`) |
| Deployment Artifacts | `pickle` |

---

## Getting Started

```bash
# Clone the repository
git clone <repo-url>
cd "ML Projects"

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras xgboost opencv-python pytesseract

# Open any notebook
jupyter notebook
```

---

## Author

**Keertan Sharma**
