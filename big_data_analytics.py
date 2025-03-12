import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import logging
import sys
import os

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BigDataAnalytics:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None

    def load_data(self):
        """Loads data from the specified path."""
        logging.info(f"Loading data from {self.data_path}")
        if os.path.exists(self.data_path):
            self.data = pd.read_csv(self.data_path)
            logging.info("Data loaded successfully.")
        else:
            logging.error("Data file does not exist.")
            sys.exit(1)

    def preprocess_data(self):
        """Preprocess the data: handle missing values, encode categorical variables."""
        logging.info("Starting data preprocessing.")
        initial_shape = self.data.shape
        self.data.dropna(inplace=True)
        logging.info(f"Dropped missing values. Initial shape: {initial_shape}, New shape: {self.data.shape}")
        
        # Encode categorical variables
        for col in self.data.select_dtypes(include=['object']).columns:
            self.data[col] = self.data[col].astype('category').cat.codes
            logging.info(f"Encoded categorical column: {col}")

    def exploratory_data_analysis(self):
        """Visualizes the data and provides a summary."""
        logging.info("Conducting exploratory data analysis (EDA).")
        sns.pairplot(self.data)
        plt.title("Pairplot of Dataset")
        plt.savefig('pairplot.png')
        plt.clf()
        logging.info("Pairplot saved as pairplot.png")
        
        correlation_matrix = self.data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title("Correlation Heatmap")
        plt.savefig('correlation_heatmap.png')
        plt.clf()
        logging.info("Correlation heatmap saved as correlation_heatmap.png")

    def split_data(self, target_variable):
        """Splits the dataset into training and testing sets."""
        logging.info(f"Splitting data with target variable: {target_variable}")
        X = self.data.drop(target_variable, axis=1)
        y = self.data[target_variable]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logging.info("Data split into train and test sets.")
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        """Trains a linear regression model."""
        logging.info("Training Linear Regression model.")
        model = LinearRegression()
        model.fit(X_train, y_train)
        logging.info("Model training completed.")
        return model

    def evaluate_model(self, model, X_test, y_test):
        """Evaluates the trained model."""
        logging.info("Evaluating model performance.")
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        logging.info(f'Mean Squared Error: {mse}')
        logging.info(f'R^2 Score: {r2}')

    def run(self, target_variable):
        """Execute the full data analysis process."""
        self.load_data()
        self.preprocess_data()
        self.exploratory_data_analysis()
        X_train, X_test, y_train, y_test = self.split_data(target_variable)
        model = self.train_model(X_train, y_train)
        self.evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    file_path = 'data/big_data_file.csv'  # Change to your CSV file path
    target_variable = 'target'  # Change to your target variable name
    analytics = BigDataAnalytics(file_path)
    analytics.run(target_variable)