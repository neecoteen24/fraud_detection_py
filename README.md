# Fraud Detection AML System

A machine learning-based fraud detection system for financial transactions, built with Streamlit for easy interaction.

## Project Overview

This system analyzes financial transactions to detect potential fraud using a trained machine learning model. The model was trained on a dataset of 6.3M+ transactions with various features including transaction type, amount, and account balances.

## Files Description

- **`fraud_detection.py`** - Main Streamlit application for fraud prediction
- **`model.ipynb`** - Jupyter notebook containing the model training process
- **`fraud_detection_pipeline.pkl`** - Trained machine learning pipeline (pickle file)
- **`AIML Dataset.csv`** - Original dataset (6.3M+ transactions)
- **`requirements.txt`** - Python dependencies

📂 Dataset Source
This project uses a financial transactions dataset sourced from Kaggle:

🔗 https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset?resource=download

You can download the dataset from the above link and place it in the project root directory as AIML Dataset.csv.

## Features

### Model Characteristics
- **Algorithm**: Logistic Regression with balanced class weights
- **Features**: Transaction type, amount, sender/receiver balances
- **Performance**: 95% accuracy, 94% recall for fraud detection
- **Training Data**: 6.3M+ transactions with 0.1% fraud rate

### Application Features
- Interactive web interface using Streamlit
- Real-time fraud prediction
- Transaction validation and balance consistency checks
- Risk indicators and warnings
- Probability scores for predictions
- Comprehensive error handling

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run fraud_detection.py
   ```

2. Open your browser and navigate to the provided URL (usually http://localhost:8501)

3. Enter transaction details:
   - Transaction Type (PAYMENT, TRANSFER, CASH_OUT, DEPOSIT)
   - Amount
   - Sender balances (old and new)
   - Receiver balances (old and new)

4. Click "Predict Fraud" to get the prediction

## Model Insights

### Fraud Patterns
- Fraud typically occurs in **TRANSFER** and **CASH_OUT** transactions
- High-risk indicators:
  - Large transaction amounts (>$100,000)
  - Account emptying after transaction
  - Transaction amount exceeding sender balance

### Data Distribution
- **Transaction Types**: PAYMENT (most common), TRANSFER, CASH_OUT, DEPOSIT
- **Fraud Rate**: 0.1% of all transactions
- **Fraud by Type**: Only TRANSFER and CASH_OUT transactions contain fraud

## Validation Features

The application includes several validation checks:
- Balance consistency validation
- Automatic correction of balance discrepancies
- Risk factor identification
- Transaction amount validation

## Technical Details

### Model Pipeline
1. **Preprocessing**: StandardScaler for numeric features, OneHotEncoder for categorical
2. **Classification**: Logistic Regression with balanced class weights
3. **Feature Engineering**: Balance differences (created but not used in final model)

### Data Preprocessing
- Removed non-essential columns (step, nameOrig, nameDest, isFlaggedFraud)
- Handled class imbalance with balanced class weights
- Applied train-test split (70-30) with stratification

## Limitations

- This is a demonstration model and should not be used in production without further validation
- The model may not capture all fraud patterns
- Performance metrics are based on the training dataset
- Real-world fraud patterns may differ from the training data

## Future Improvements

- Add more sophisticated feature engineering
- Implement ensemble methods
- Add real-time transaction monitoring
- Include additional fraud detection rules
- Add model retraining capabilities

## License

This project is for educational and demonstration purposes. 
