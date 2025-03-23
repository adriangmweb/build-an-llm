import urllib.request
import zipfile
import os
from pathlib import Path

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return

    # Downloading the file
    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())

    # Unzipping the file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    # Add .tsv file extension
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")

try:
    download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
    print(f"Primary URL failed: {e}. Trying backup URL...")
    url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip"
    download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path) 

import pandas as pd

from functions import create_balanced_dataset, random_split

df = pd.read_csv(
    "sms_spam_collection/SMSSpamCollection.tsv", 
    sep="\t", 
    header=None, 
    names=["Label", "Text"]
)

print("Original dataset:")
print(df["Label"].value_counts())
print("\n")

print("Creating balanced dataset...")
balanced_df = create_balanced_dataset(df)
print(balanced_df["Label"].value_counts())

# Convert the labels to integers
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

# Create train and test splits
train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)

# Save the dataset to csv files
train_df.to_csv("train_dataset.csv", index=None)
validation_df.to_csv("validation_dataset.csv", index=None)
test_df.to_csv("test_dataset.csv", index=None)
