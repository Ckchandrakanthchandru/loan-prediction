
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

df = pd.read_csv(r'C:\Users\Sujitha\OneDrive\Desktop\Ml Intern\train_u6lujuX_CVtuZ9i.csv')

print(df.head())
# %%
print(df.info())
# %%
print(df.describe())
# %%
print(df.isnull().sum())
# %%
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
print(df.isnull().sum())

df['LoanAmount'].fillna(df['LoanAmount'].mode()[0], inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
# %%
print(df.isnull().sum())

sns.countplot(x='Loan_Status', data=df)
plt.show()

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

df = pd.get_dummies(df, columns=['Property_Area', 'Dependents'], drop_first=True)

# %%

X_cls = df.drop(['Loan_ID', 'Loan_Status'], axis=1)
y_cls = df['Loan_Status']

Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

#%%
#Logistic Regression
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(Xc_train, yc_train)


#%%
#Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier()
dt_model.fit(Xc_train, yc_train)

#Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier()
rf_model.fit(Xc_train, yc_train)


#%%
#SVM
from sklearn.svm import SVC
svm_model = SVC()
svm_model.fit(Xc_train, yc_train)

#KNN
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier()
knn_model.fit(Xc_train, yc_train)

# %%
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

models = {
    "Logistic Regression": log_model,
    "Decision Tree": dt_model,
    "Random Forest": rf_model,
    "SVM": svm_model,
    "KNN": knn_model
}

for name, model in models.items():
    y_pred = model.predict(Xc_test)
    print(f"\n🔍 {name}")
    print("Accuracy:", accuracy_score(yc_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(yc_test, y_pred))
    print("Classification Report:\n", classification_report(yc_test, y_pred))

# %%
df_reg = df[df['LoanAmount'] > 0]  # just to be safe

X_reg = df_reg.drop(['Loan_ID', 'LoanAmount'], axis=1)
y_reg = df_reg['LoanAmount']

Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

models_r = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "SVM": SVR(),
    "KNN": KNeighborsRegressor()
}

for name, model in models_r.items():
    model.fit(Xr_train, yr_train)


# %%
from sklearn.metrics import mean_squared_error, r2_score

for name, model in models_r.items():
    y_pred = model.predict(Xr_test)
    print(f"\n🔍 {name}")
    print("RMSE:", np.sqrt(mean_squared_error(yr_test, y_pred)))
    print("R2 Score:", r2_score(yr_test, y_pred))


#TEST DATA

import pandas as pd

# Load the test CSV file
test_df = pd.read_csv(r"C:\Users\Sujitha\OneDrive\Desktop\Ml Intern\test_Y3wMUE5_7gLdaTN.csv")


# Backup Loan_ID to reattach later
loan_ids = test_df['Loan_ID']

# Fill missing values (same as training)
test_df['Gender'].fillna(test_df['Gender'].mode()[0], inplace=True)
test_df['Married'].fillna(test_df['Married'].mode()[0], inplace=True)
test_df['Dependents'].fillna(test_df['Dependents'].mode()[0], inplace=True)
test_df['Self_Employed'].fillna(test_df['Self_Employed'].mode()[0], inplace=True)
test_df['LoanAmount'].fillna(test_df['LoanAmount'].median(), inplace=True)
test_df['Loan_Amount_Term'].fillna(test_df['Loan_Amount_Term'].mode()[0], inplace=True)
test_df['Credit_History'].fillna(test_df['Credit_History'].mode()[0], inplace=True)

# Encode binary categorical values
test_df['Gender'] = test_df['Gender'].map({'Male': 1, 'Female': 0})
test_df['Married'] = test_df['Married'].map({'Yes': 1, 'No': 0})
test_df['Education'] = test_df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
test_df['Self_Employed'] = test_df['Self_Employed'].map({'Yes': 1, 'No': 0})

# One-hot encode Property_Area and Dependents
test_df = pd.get_dummies(test_df, columns=['Property_Area', 'Dependents'], drop_first=True)

test_df.drop('Loan_ID', axis=1, inplace=True)

#PREDICT

# Align test data columns with training data
test_df = test_df.reindex(columns=X_cls.columns, fill_value=0)

# Predict using the best classification model (e.g., Random Forest)
test_predictions = rf_model.predict(test_df)

# If you want to convert predictions back to Y/N format
test_predictions = ['Y' if pred == 1 else 'N' for pred in test_predictions]

# Combine with Loan_ID and save to CSV
submission = pd.DataFrame({
    'Loan_ID': loan_ids,
    'Loan_Status': test_predictions
})

submission.to_csv("loan_approval_predictions.csv", index=False)
print("✅ Loan approval predictions saved as 'loan_approval_predictions.csv'")


# Align test data with regression input columns
test_df_reg = test_df.reindex(columns=X_reg.columns, fill_value=0)

# Predict loan amount using the best regression model (e.g., Random Forest)
loan_amount_preds = models_r["Random Forest"].predict(test_df_reg)

# Save to CSV
reg_output = pd.DataFrame({
    'Loan_ID': loan_ids,
    'Predicted_LoanAmount': loan_amount_preds
})

reg_output.to_csv("predicted_loan_amounts.csv", index=False)
print("✅ Loan amount predictions saved as 'predicted_loan_amounts.csv'")
