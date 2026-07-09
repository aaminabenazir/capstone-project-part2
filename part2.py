import pandas as pd
import os
import numpy as np
os.chdir('/Users/aaminabenazir/Desktop/')
df=pd.read_csv('cleaned_data.csv')
print(df.head())
# 1. Define feature matrix X (drop the regression target column)
# Replace 'your_target_column_name' with the actual name of your continuous target
target_col = 'median_house_value' 
X = df.drop(columns=[target_col])

# 2. Define regression label
y_reg = df[target_col]

# 3. Define classification label (binarized at the median)
# This creates 0 for below median, 1 for above median
y_clf = (y_reg > y_reg.median()).astype(int)
# Example of One-Hot Encoding for nominal columns
# 'drop_first=True' is used to prevent multicollinearity
X = pd.get_dummies(X, drop_first=True)

# Example of Label Encoding for ordinal columns (if you have them)
# mapping = {'Low': 0, 'Medium': 1, 'High': 2}
# X['ordinal_column'] = X['ordinal_column'].map(mapping)
from sklearn.model_selection import train_test_split

# Split X and both labels
# We use the same random_state to ensure X_train and X_test correspond to the same rows for both labels
X_train, X_test, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
_, _, y_clf_train, y_clf_test = train_test_split(X, y_clf, test_size=0.2, random_state=42)
from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# Fit only on the training set
# This calculates the mean and standard deviation of the training data
scaler.fit(X_train)

# Transform both sets using the statistics from the training set
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Initialize and train the model
lr = LinearRegression()
lr.fit(X_train_scaled, y_reg_train)

# Predict on the test set
y_pred_lr = lr.predict(X_test_scaled)

# Calculate metrics
mse_lr = mean_squared_error(y_reg_test, y_pred_lr)
r2_lr = r2_score(y_reg_test, y_pred_lr)

print(f"Linear Regression - MSE: {mse_lr:.4f}, R2: {r2_lr:.4f}")
from sklearn.linear_model import Ridge

# Initialize and train with alpha=1.0
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_reg_train)

# Predict
y_pred_ridge = ridge.predict(X_test_scaled)

# Calculate metrics
mse_ridge = mean_squared_error(y_reg_test, y_pred_ridge)
r2_ridge = r2_score(y_reg_test, y_pred_ridge)

print(f"Ridge Regression - MSE: {mse_ridge:.4f}, R2: {r2_ridge:.4f}")
# Create a series of coefficients
coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': lr.coef_})
coef_df['Abs_Coef'] = coef_df['Coefficient'].abs()

# Sort and print top 3
top_3 = coef_df.sort_values(by='Abs_Coef', ascending=False).head(3)
print(top_3)
# Check imbalance
print(y_clf_train.value_counts(normalize=True))

# Train Logistic Regression with balanced class weights
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Using class_weight='balanced' automatically adjusts weights inversely proportional to class frequencies
clf = LogisticRegression(max_iter=1000, class_weight='balanced')
clf.fit(X_train_scaled, y_clf_train)
y_pred_clf = clf.predict(X_test_scaled)
y_prob_clf = clf.predict_proba(X_test_scaled)[:, 1] # Probabilities for the positive class

# Print report
print(classification_report(y_clf_test, y_pred_clf))

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_clf_test, y_prob_clf)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_clf_test, y_prob_clf):.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
from sklearn.metrics import precision_score, recall_score, f1_score

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
results = []

for t in thresholds:
    # Create predictions based on the custom threshold
    y_pred_t = (y_prob_clf >= t).astype(int)
    results.append({
        'Threshold': t,
        'Precision': precision_score(y_clf_test, y_pred_t),
        'Recall': recall_score(y_clf_test, y_pred_t),
        'F1': f1_score(y_clf_test, y_pred_t)
    })

results_df = pd.DataFrame(results)
print(results_df)
# Train second model with strong regularization (C=0.01)
clf_reg = LogisticRegression(max_iter=1000, C=0.01, class_weight='balanced')
clf_reg.fit(X_train_scaled, y_clf_train)

# Predict probabilities for the new model
y_prob_reg = clf_reg.predict_proba(X_test_scaled)[:, 1]

# Compare AUCs
auc_base = roc_auc_score(y_clf_test, y_prob_clf)
auc_reg = roc_auc_score(y_clf_test, y_prob_reg)

print(f"Baseline (C=1.0) AUC: {auc_base:.4f}")
print(f"Regularized (C=0.01) AUC: {auc_reg:.4f}")
n_iterations = 500
auc_diffs = []

# Get predictions for both models
y_prob_base = clf.predict_proba(X_test_scaled)[:, 1]
y_prob_reg = clf_reg.predict_proba(X_test_scaled)[:, 1]

# Convert arrays to pandas for easier indexing
y_test_arr = y_clf_test.values

for i in range(n_iterations):
    # Sample indices with replacement
    indices = np.random.choice(len(y_test_arr), size=len(y_test_arr), replace=True)
    
    # Calculate AUC for this bootstrap sample
    auc_base_sample = roc_auc_score(y_test_arr[indices], y_prob_base[indices])
    auc_reg_sample = roc_auc_score(y_test_arr[indices], y_prob_reg[indices])
    
    auc_diffs.append(auc_base_sample - auc_reg_sample)

# Calculate 2.5th and 97.5th percentiles
lower = np.percentile(auc_diffs, 2.5)
upper = np.percentile(auc_diffs, 97.5)

print(f"95% Confidence Interval for AUC difference: ({lower:.4f}, {upper:.4f})")
