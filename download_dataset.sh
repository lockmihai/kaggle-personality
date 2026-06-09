#!/bin/bash

# Create data directory if it doesn't exist
mkdir -p data

# Check if kaggle CLI is installed
if ! command -v kaggle &> /dev/null
then
    echo "Kaggle CLI is not installed or not in PATH."
    echo "Please install it via 'pip install kaggle' and set up your API credentials (~/.kaggle/kaggle.json)."
    echo "Alternatively, download the dataset manually from:"
    echo "https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis"
    echo "Extract 'marketing_campaign.csv' and place it in the 'data/' directory."
    exit 1
fi

echo "Downloading dataset using Kaggle CLI..."
kaggle datasets download -d imakash3011/customer-personality-analysis -p data/ --unzip

if [ $? -eq 0 ]; then
    echo "Dataset downloaded and extracted successfully to data/"
else
    echo "Failed to download dataset. Check your Kaggle credentials."
fi
