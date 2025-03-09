# filter_and_save_external_data.py

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics.pairwise import cosine_similarity
import datasets

# Function to convert SMILES to Morgan fingerprint
def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))
    return None

# Function to compute similarity matrix and select relevant data
def filter_external_data(target_smiles, external_data, similarity_threshold=0.7):
    # Convert target dataset SMILES to fingerprints
    target_fps = [smiles_to_fingerprint(smiles) for smiles in target_smiles]
    
    # Convert external dataset SMILES to fingerprints
    external_fps = [smiles_to_fingerprint(smiles) for smiles in external_data['SMILES']]

    # Filter out None fingerprints
    valid_target_fps = [fp for fp in target_fps if fp is not None]
    valid_external_fps = [fp for fp in external_fps if fp is not None]

    if not valid_target_fps or not valid_external_fps:
        raise ValueError("No valid fingerprints found. Check SMILES conversion.")

    # Compute similarity matrix
    similarity_matrix = cosine_similarity(valid_target_fps, valid_external_fps)

    # Select external data points that have high similarity to the target dataset
    selected_indices = np.where(similarity_matrix.max(axis=0) >= similarity_threshold)[0]
    filtered_external_data = external_data.iloc[selected_indices]

    return filtered_external_data

# Load the dataset
DATASET_PATH = "scikit-fingerprints/MoleculeNet_Lipophilicity"
dataset = datasets.load_dataset(DATASET_PATH)
dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)

# Load the external dataset
external_data = pd.read_csv('/home/neuronet_team146/Project_Files/scripts/External_Dataset_for_Task2.csv')

# Get the target SMILES from the training dataset
target_smiles = dataset['train']['SMILES']

# Filter external data based on similarity with target dataset
filtered_external_data = filter_external_data(target_smiles, external_data)
print(f"====================== Cosine filtered external data length : - ", len(filtered_external_data))

# Save filtered external data to a new CSV file
filtered_external_data.to_csv('/home/neuronet_team146/Project_Files/scripts/Cosine_Similarity_Filtered_Dataset.csv', index=False)

print(f"Filtered external data saved to '/home/neuronet_team146/Project_Files/scripts/Cosine_Similarity_Filtered_Dataset.csv'")

# create_train_test_loader.py

# import datasets
# from transformers import AutoTokenizer
# import torch
# from torch.utils.data import DataLoader, TensorDataset
# import pandas as pd
# import warnings

# # Suppress deprecation warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)

# # Dataset and model setup
# DATASET_PATH = "scikit-fingerprints/MoleculeNet_Lipophilicity"
# MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"

# # Tokenizer setup
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# def tokenize_function(example):
#     return tokenizer(example['SMILES'], truncation=True, padding="max_length", max_length=128)

# # Load original dataset
# dataset = datasets.load_dataset(DATASET_PATH)
# dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)

# # Tokenize train and test datasets
# train_data = dataset['train'].map(tokenize_function, batched=True)
# test_data = dataset['test'].map(tokenize_function, batched=True)

# # Load the filtered external dataset
# filtered_external_data = pd.read_csv('/home/neuronet_team146/Project_Files/scripts/Cosine_Similarity_Filtered_Dataset.csv')

# # Load the original train data and standardize column names for merging
# train_data_df = dataset['train'].to_pandas()
# filtered_external_data.rename(columns={'Label': 'label'}, inplace=True)

# # Combine the target dataset with the filtered external data
# combined_data = pd.concat([train_data_df[['SMILES', 'label']], filtered_external_data[['SMILES', 'label']]], ignore_index=True)

# # Print dataset sizes
# print(f"Original Train Data: {len(train_data_df)}")
# print(f"Filtered External Data: {len(filtered_external_data)}")
# print(f"Combined Data: {len(combined_data)}")

# # Tokenization for combined dataset
# tokenized_combined_data = tokenizer(list(combined_data['SMILES']), truncation=True, padding="max_length", max_length=128, return_tensors="pt")

# # Convert to PyTorch tensors
# def convert_to_torch(tokenized_data, labels):
#     input_ids = tokenized_data['input_ids']
#     attention_mask = tokenized_data['attention_mask']
#     labels = torch.tensor(labels, dtype=torch.float)
#     return TensorDataset(input_ids, attention_mask, labels)

# # Convert combined data to torch
# train_dataset = convert_to_torch(tokenized_combined_data, combined_data['label'].tolist())

# # For test data, convert the tokenized data from Hugging Face dataset
# tokenized_test_data = tokenizer(list(dataset['test']['SMILES']), truncation=True, padding="max_length", max_length=128, return_tensors="pt")

# # Ensure that 'label' exists in the dataset
# test_labels = dataset['test']['label']

# # Convert test data to torch
# test_dataset = convert_to_torch(tokenized_test_data, test_labels)

# # Create DataLoader for training and testing
# train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# # Print out dataloader sizes
# print(f"Train Dataloader Size: {len(train_dataloader.dataset)}")
# print(f"Test Dataloader Size: {len(test_dataloader.dataset)}")

