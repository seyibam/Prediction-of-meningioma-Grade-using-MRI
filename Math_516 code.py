# 1. Preliminary data analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('C:/Users/USER/Desktop/Dataset/Assign 516/Dataset-_01.csv')

# Plotting the distribution of 'Grade'
plt.figure(figsize=(8, 6))
sns.countplot(x='Grade', data=data)
plt.title('Distribution of Meningioma Grades')
plt.xlabel('Meningioma Grade')
plt.ylabel('Count')
plt.show()

import pandas as pd

# Load the dataset
data = pd.read_csv('C:/Users/USER/Desktop/Dataset/Assign 516/Dataset-_01.csv')

# Convert 'Grade' to a numerical format
if data['Grade'].dtype == 'object':
    # Creating a simple mapping based on unique values sorted alphabetically
    grade_mapping = {grade: idx for idx, grade in enumerate(sorted(data['Grade'].unique()))}
    data['Grade'] = data['Grade'].map(grade_mapping)

# Selecting only numeric columns
numeric_data = data.select_dtypes(include=[np.number])

# Calculate IQR for each numeric column and detect outliers
outliers_count = 0
for column in numeric_data.columns:
    Q1 = numeric_data[column].quantile(0.25)
    Q3 = numeric_data[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers = numeric_data[(numeric_data[column] < (Q1 - 1.5 * IQR)) | (numeric_data[column] > (Q3 + 1.5 * IQR))]
    outliers_count += outliers.shape[0]

print(f'Total number of outliers detected: {outliers_count}')

# PCA Visualization
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardizing data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.select_dtypes(include=[np.number]))

pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_data)
pca_df = pd.DataFrame(data=pca_components, columns=['PC1', 'PC2'])
pca_df['Grade'] = data['Grade']

sns.scatterplot(x='PC1', y='PC2', hue='Grade', data=pca_df)
plt.title('PCA Plot')
plt.show()

import pandas as pd

# Load the dataset
data = pd.read_csv('C:/Users/USER/Desktop/Dataset/Assign 516/Dataset-_01.csv')

# Convert 'Grade' to a numerical format if it's categorical
if data['Grade'].dtype == 'object':
    # Creating a simple mapping based on unique values sorted alphabetically
    # If the grades are like 'Grade I', 'Grade II', sort them correctly as required
    grade_mapping = {grade: idx for idx, grade in enumerate(sorted(data['Grade'].unique()))}
    data['Grade'] = data['Grade'].map(grade_mapping)

# Select only numeric data for correlation analysis
numeric_data = data.select_dtypes(include=[np.number])

# Calculate the correlation matrix
correlation_matrix = numeric_data.corr()

# Output the correlation matrix
print(correlation_matrix)

# To view strong correlations, you can filter the matrix
strong_correlations = correlation_matrix[correlation_matrix.abs() > 0.5]
print("Strong Correlations:\n", strong_correlations.fillna(0))


# 2. Data standardisation and Feature selection
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('C:/Users/USER/Desktop/Dataset/Assign 516/standardized_dataset.csv')

# Separate features and target
X = data.iloc[:, 2:]  # Features
y = data.iloc[:, 1]  # Target is the 'Grade' column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and apply the StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled arrays back to DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

# Feature selection - Filter method
selector = SelectKBest(f_classif, k=10)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)  # Apply the same transformation to test data
selected_features_filter = X_train_scaled.columns[selector.get_support()]

# Feature selection - LASSO
lasso = LassoCV(cv=5).fit(X_train_scaled, y_train)
selected_features_lasso = X_train_scaled.columns[(lasso.coef_ != 0)]
X_test_lasso = X_test_scaled[selected_features_lasso]

# Export selected features to CSV for both training and testing datasets
# Filter method
X_train_filter = X_train_scaled[selected_features_filter]
X_train_filter['Grade'] = y_train.reset_index(drop=True)
X_train_filter.to_csv('C:/Users/USER/Desktop/Dataset/Assign 516/X_train_filter_selected.csv', index=False)

X_test_filter = X_test_scaled[selected_features_filter]
X_test_filter['Grade'] = y_test.reset_index(drop=True)
X_test_filter.to_csv('C:/Users/USER/Desktop/Dataset/Assign 516/X_test_filter_selected.csv', index=False)

# LASSO method
X_train_lasso = X_train_scaled[selected_features_lasso]
X_train_lasso['Grade'] = y_train.reset_index(drop=True)
X_train_lasso.to_csv('C:/Users/USER/Desktop/Dataset/Assign 516/X_train_lasso_selected.csv', index=False)

X_test_lasso['Grade'] = y_test.reset_index(drop=True)
X_test_lasso.to_csv('C:/Users/USER/Desktop/Dataset/Assign 516/X_test_lasso_selected.csv', index=False)

print("Training and test data with selected features saved successfully.")

# 3. Model Training and Model evaluation
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Function to load data, split it into train and test sets, and standardize it
def prepare_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop('Grade', axis=1)
    y = data['Grade']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# Function to train and evaluate a model
def train_evaluate_model(model, params, X_train, y_train, X_test, y_test):
    grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return best_model, accuracy, report

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'logreg': {'C': [0.1, 1, 10]},
    'rf': {'max_depth': [10, 20, None], 'n_estimators': [50, 100, 200]},
    'svc': {'C': [0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1], 'kernel': ['rbf']}
}

# Paths to the datasets
filter_file_path = 'C:/Users/USER/Desktop/Dataset/Assign 516/X_train_filter_selected.csv'
lasso_file_path = 'C:/Users/USER/Desktop/Dataset/Assign 516/X_train_lasso_selected.csv'

# Prepare data for filter-selected and LASSO-selected features
X_train_filter, X_test_filter, y_train_filter, y_test_filter = prepare_data(filter_file_path)
X_train_lasso, X_test_lasso, y_train_lasso, y_test_lasso = prepare_data(lasso_file_path)

# Initialize the classifiers
logreg = LogisticRegression(random_state=42)
rf = RandomForestClassifier(random_state=42)
svc = SVC(random_state=42)

# Train and evaluate models on filter-selected dataset
print("Training on filter-selected features:")
logreg_model_filter, logreg_accuracy_filter, logreg_report_filter = train_evaluate_model(
    logreg, param_grid['logreg'], X_train_filter, y_train_filter, X_test_filter, y_test_filter
)
rf_model_filter, rf_accuracy_filter, rf_report_filter = train_evaluate_model(
    rf, param_grid['rf'], X_train_filter, y_train_filter, X_test_filter, y_test_filter
)
svc_model_filter, svc_accuracy_filter, svc_report_filter = train_evaluate_model(
    svc, param_grid['svc'], X_train_filter, y_train_filter, X_test_filter, y_test_filter
)

# Train and evaluate models on LASSO-selected dataset
print("Training on LASSO-selected features:")
logreg_model_lasso, logreg_accuracy_lasso, logreg_report_lasso = train_evaluate_model(
    logreg, param_grid['logreg'], X_train_lasso, y_train_lasso, X_test_lasso, y_test_lasso
)
rf_model_lasso, rf_accuracy_lasso, rf_report_lasso = train_evaluate_model(
    rf, param_grid['rf'], X_train_lasso, y_train_lasso, X_test_lasso, y_test_lasso
)
svc_model_lasso, svc_accuracy_lasso, svc_report_lasso = train_evaluate_model(
    svc, param_grid['svc'], X_train_lasso, y_train_lasso, X_test_lasso, y_test_lasso
)

# Print the evaluation results for both datasets
print("\nFilter-selected Features:")
print("Logistic Regression Accuracy:", logreg_accuracy_filter)
print(logreg_report_filter)
print("Random Forest Accuracy:", rf_accuracy_filter)
print(rf_report_filter)
print("SVC Accuracy:", svc_accuracy_filter)
print(svc_report_filter)

print("\nLASSO-selected Features:")
print("Logistic Regression Accuracy:", logreg_accuracy_lasso)
print(logreg_report_lasso)
print("Random Forest Accuracy:", rf_accuracy_lasso)
print(rf_report_lasso)
print("SVC Accuracy:", svc_accuracy_lasso)
print(svc_report_lasso)

import joblib

# Saving models trained on filter-selected features
joblib.dump(logreg_model_filter, 'C:/Users/USER/Desktop/Dataset/Assign 516/logreg_model_filter.pkl')
joblib.dump(rf_model_filter, 'C:/Users/USER/Desktop/Dataset/Assign 516/rf_model_filter.pkl')
joblib.dump(svc_model_filter, 'C:/Users/USER/Desktop/Dataset/Assign 516/svc_model_filter.pkl')

# Saving models trained on LASSO-selected features
joblib.dump(logreg_model_lasso, 'C:/Users/USER/Desktop/Dataset/Assign 516/logreg_model_lasso.pkl')
joblib.dump(rf_model_lasso, 'C:/Users/USER/Desktop/Dataset/Assign 516/rf_model_lasso.pkl')
joblib.dump(svc_model_lasso, 'C:/Users/USER/Desktop/Dataset/Assign 516/svc_model_lasso.pkl')

print("All models have been saved successfully.")

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import label_binarize
import numpy as np
import joblib  # for loading model files

# Function to load data
def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop('Grade', axis=1)
    y = data['Grade']
    return X, y

# Function to calculate multiple evaluation metrics
def evaluate_model(y_test, y_pred, model, X_test):
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1 Score': f1_score(y_test, y_pred, average='weighted')
    }

    # Calculate ROC-AUC if the model supports probability predictions
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
        if len(np.unique(y_test)) == 2:  # Binary classification
            metrics['ROC AUC'] = roc_auc_score(y_test, y_prob[:, 1])
        else:  # Multi-class classification
            y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
            metrics['ROC AUC'] = roc_auc_score(y_test_bin, y_prob, average='weighted', multi_class='ovr')

    return metrics

# Load models for both filter-selected and LASSO-selected features
logreg_model_filter = joblib.load('C:/Users/USER/Desktop/Dataset/Assign 516/logreg_model_filter.pkl')
rf_model_filter = joblib.load('C:/Users/USER/Desktop/Dataset/Assign 516/rf_model_filter.pkl')
svc_model_filter = joblib.load('C:/Users/USER/Desktop/Dataset/Assign 516/svc_model_filter.pkl')

logreg_model_lasso = joblib.load('C:/Users/USER/Desktop/Dataset/Assign 516/logreg_model_lasso.pkl')
rf_model_lasso = joblib.load('C:/Users/USER/Desktop/Dataset/Assign 516/rf_model_lasso.pkl')
svc_model_lasso = joblib.load('C:/Users/USER/Desktop/Dataset/Assign 516/svc_model_lasso.pkl')

# Paths to the test datasets
test_filter_path = 'C:/Users/USER/Desktop/Dataset/Assign 516/X_test_filter_selected.csv'
test_lasso_path = 'C:/Users/USER/Desktop/Dataset/Assign 516/X_test_lasso_selected.csv'

# Load test data
X_test_filter, y_test_filter = load_data(test_filter_path)
X_test_lasso, y_test_lasso = load_data(test_lasso_path)

# Evaluate models on filter-selected test data
print("Evaluation on Filter-selected Features:")
for model, name in zip([logreg_model_filter, rf_model_filter, svc_model_filter], ['Logistic Regression', 'Random Forest', 'SVC']):
    y_pred = model.predict(X_test_filter)
    metrics = evaluate_model(y_test_filter, y_pred, model, X_test_filter)
    print(f"\n{name}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

# Evaluate models on LASSO-selected test data
print("\nEvaluation on LASSO-selected Features:")
for model, name in zip([logreg_model_lasso, rf_model_lasso, svc_model_lasso], ['Logistic Regression', 'Random Forest', 'SVC']):
    y_pred = model.predict(X_test_lasso)
    metrics = evaluate_model(y_test_lasso, y_pred, model, X_test_lasso)
    print(f"\n{name}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

# 4.Hyperparameter tuning and model evaluation
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def load_and_prepare_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop('Grade', axis=1)
    y = data['Grade']
    return train_test_split(X, y, test_size=0.3, random_state=42)

def standardize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def hyperparameter_tuning(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', verbose=1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

# Path to the datasets
filter_dataset_path = 'C:/Users/USER/Desktop/Dataset/Assign 516/X_train_filter_selected.csv'
lasso_dataset_path = 'C:/Users/USER/Desktop/Dataset/Assign 516/X_train_lasso_selected.csv'

# Parameter grid for hyperparameter tuning
param_grid = {
    'LogisticRegression': {'C': [0.1, 1, 10]},
    'RandomForestClassifier': {'max_depth': [10, 20, None], 'n_estimators': [50, 100, 200]},
    'SVC': {'C': [0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1], 'kernel': ['rbf']}
}

# Load, prepare, and standardize data for both feature selection methods
X_train_filter, X_test_filter, y_train_filter, y_test_filter = load_and_prepare_data(filter_dataset_path)
X_train_filter_scaled, X_test_filter_scaled = standardize_data(X_train_filter, X_test_filter)
X_train_lasso, X_test_lasso, y_train_lasso, y_test_lasso = load_and_prepare_data(lasso_dataset_path)
X_train_lasso_scaled, X_test_lasso_scaled = standardize_data(X_train_lasso, X_test_lasso)

# Initialize classifiers
models = {
    'LogisticRegression': LogisticRegression(random_state=42),
    'RandomForestClassifier': RandomForestClassifier(random_state=42),
    'SVC': SVC(random_state=42, probability=True)
}

# Apply hyperparameter tuning and evaluation for each feature selection method
for feature_set, data in [('Filter-selected', (X_train_filter_scaled, X_test_filter_scaled, y_train_filter, y_test_filter)),
                          ('LASSO-selected', (X_train_lasso_scaled, X_test_lasso_scaled, y_train_lasso, y_test_lasso))]:
    print(f"\nTraining on {feature_set} features:")
    X_train, X_test, y_train, y_test = data
    for name, model in models.items():
        print(f"\nHyperparameter tuning for {name}:")
        tuned_model = hyperparameter_tuning(model, param_grid[name], X_train, y_train)  # Correct key access
        accuracy, report = evaluate_model(tuned_model, X_test, y_test)
        print(f"\n{name} - {feature_set} Evaluation:")
        print(f"Accuracy: {accuracy}")
        print(f"Classification Report:\n{report}")

