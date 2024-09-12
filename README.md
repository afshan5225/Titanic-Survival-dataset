# Titanic Survival Prediction using Logistic Regression

## Problem Statement

The Titanic dataset is a classic dataset used in machine learning and data science to predict the survival of passengers based on various features such as age, sex, and class. The goal of this project is to build a Logistic Regression model to predict whether a passenger survived the Titanic disaster based on the available features.

## Libraries Used

This project uses the following Python libraries:

- `pandas` - For data manipulation and analysis.
- `numpy` - For numerical operations.
- `matplotlib.pyplot` - For data visualization.
- `seaborn` - For statistical data visualization.
- `sklearn` - For machine learning algorithms and metrics.
- `pickle` - For model serialization and deserialization.

## Execution Methodology

1. **Data Import and Exploration**:
   - Load the Titanic dataset from a URL.
   - Explore the dataset to understand its structure and identify missing values.

2. **Data Cleaning**:
   - Handle missing values in `Age` and `Embarked` columns.
   - Drop unnecessary columns such as `Cabin`, `PassengerId`, `Name`, and `Ticket`.

3. **Feature Engineering**:
   - Encode categorical features (`Sex` and `Embarked`) using `LabelEncoder`.
   - Convert `Age` to integer and round `Fare` values.

4. **Exploratory Data Analysis (EDA)**:
   - Visualize the relationship between features and survival rates.
   - Analyze feature importance using `ExtraTreesClassifier`.

5. **Model Training and Evaluation**:
   - Split the data into training and testing sets using `StratifiedKFold`.
   - Train a Logistic Regression model on the training data.
   - Evaluate the model performance using confusion matrix, accuracy score, and classification report.

6. **Model Export**:
   - Serialize the trained model using `pickle` for future use.


