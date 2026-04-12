import pandas as pd
import os
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))
repo_id = "gowdhamankarthikeyan/tourism-dataset"
DATASET_PATH = f"hf://datasets/{repo_id}/tourism.csv"

print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)

# Drop unique identifier column
df.drop(columns=['CustomerID'], inplace=True, errors='ignore')

# Handle missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

# Encode categorical columns
df = pd.get_dummies(df, drop_first=True)

# Define target variable
target_col = 'ProdTaken'
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42)

# Save locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

print("Uploading prepared data to Hugging Face...")
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id=repo_id,
        repo_type="dataset",
    )
print("Data Preparation Complete.")
