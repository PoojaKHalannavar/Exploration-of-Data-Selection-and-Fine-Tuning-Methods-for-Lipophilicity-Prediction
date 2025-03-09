"""# Task 1: Fine-tune Chemical Language Model

The goal is to fine-tune a pre-trained chemical language model on a regression task using the Lipophilicity dataset. The task involves predicting the lipophilicity value for a given molecule representation (SMILES string). You will learn how to load and tokenize a dataset from HuggingFace, how to load a pre-trained language model, and finally, how to run a model in inference mode.

Your task is to complete the missing code blocks below.
"""

# import dependencies

import torch
from datasets import load_dataset
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm.notebook import tqdm
import random

"""# 1.Fine-tune a Chemical Language Model on Lipophilicity

## --- Step 1: Load Dataset ---

The dataset we are going to use is the [Lipophilicity](https://huggingface.co/datasets/scikit-fingerprints/MoleculeNet_Lipophilicity) dataset, part of [MoleculeNet](https://pubs.rsc.org/en/content/articlelanding/2018/sc/c7sc02664a) benchmark.

Lipophilicity, also known as hydrophobicity, is a measure of how readily a substance dissolves in nonpolar solvents (such as oil) compared to polar solvents (such as water).
"""

# specify dataset name and model name

DATASET_PATH = "scikit-fingerprints/MoleculeNet_Lipophilicity"
MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"  #MoLFormer model

# load the dataset from HuggingFace

dataset = load_dataset(DATASET_PATH)

# Explore the dataset
# For example, print the column names and display a few sample rows
# TODO: your code goes here

print("Dataset column names:", dataset['train'].column_names)
print("Sample rows from the dataset:")

for i in range(5):
    print(dataset['train'][i])

# define a PyTorch Dataset class for handling SMILES strings and targets

class SMILESDataset(Dataset):
    def __init__(self, dataset):
        """
        Args:
            dataset: Hugging Face dataset containing SMILES strings and labels
        """
        self.smiles = dataset['SMILES']
        self.labels = dataset['label']

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        """
        Returns a tuple (SMILES string, label) for a given index
        """
        return self.smiles[idx], torch.tensor(self.labels[idx], dtype=torch.float)


train_dataset = SMILESDataset(dataset['train'])

"""## --- Step 2: Split Dataset ---

As there is only one split (train split) in the original dataset, we need to split the data into training and testing sets by ourselves.
"""

# tokenize the data
# load a pre-trained tokenizer from HuggingFace
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# split the data into training and test datasets
dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)
# TODO: your code goes here

def tokenize_function(examples):
    return tokenizer(examples['SMILES'], padding='max_length', truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['SMILES'])


def convert_labels(example):
    example['label'] = float(example['label'])
    return example

tokenized_datasets = tokenized_datasets.map(convert_labels)

print(tokenized_datasets['train'][0])

# construct Pytorch data loaders for both train and test datasets

# TODO: your code goes here
from torch.utils.data import TensorDataset

BATCH_SIZE = 16
def convert_to_torch(dataset):
    input_ids = torch.tensor(dataset['input_ids'])
    attention_mask = torch.tensor(dataset['attention_mask'])
    labels = torch.tensor(dataset['label'], dtype=torch.float)

    return TensorDataset(input_ids, attention_mask, labels)

train_data = convert_to_torch(tokenized_datasets['train'])
test_data = convert_to_torch(tokenized_datasets['test'])

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

for batch in train_dataloader:
    input_ids_batch, attention_mask_batch, labels_batch = batch
    print("Input IDs Shape:", input_ids_batch.shape)
    print("Attention Mask Shape:", attention_mask_batch.shape)
    print("Labels Shape:", labels_batch.shape)
    break

"""## --- Step 3: Load Model ---"""

# load pre-trained model from HuggingFace
model = AutoModel.from_pretrained(MODEL_NAME, deterministic_eval=True, trust_remote_code=True)

# We need to add a regression head on the language model as we are doing a regression task.

class MoLFormerWithRegressionHead(nn.Module):
    def __init__(self, model_name):
        super(MoLFormerWithRegressionHead, self).__init__()
        self.molformer = AutoModel.from_pretrained(model_name, deterministic_eval=True, trust_remote_code=True)


        self.regression_head = nn.Sequential(
            nn.Linear(self.molformer.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.molformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        regression_output = self.regression_head(pooled_output)
        return regression_output.squeeze(-1)


model = MoLFormerWithRegressionHead(MODEL_NAME)
print(model)

# initialize the regression model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
regression_model = MoLFormerWithRegressionHead(MODEL_NAME).to(device)

"""## --- Step 4: Training ---"""

import torch.optim as optim


criterion = nn.MSELoss()

optimizer = optim.AdamW(regression_model.parameters(), lr=5e-5, weight_decay=1e-2)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

regression_model.to(device)

EPOCHS = 5


for epoch in range(EPOCHS):
    regression_model.train()
    running_loss = 0.0

    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):

        input_ids_batch, attention_mask_batch, labels_batch = batch

        input_ids = input_ids_batch.to(device)
        attention_mask = attention_mask_batch.to(device)
        labels = labels_batch.to(device)

        optimizer.zero_grad()

        outputs = regression_model(input_ids=input_ids, attention_mask=attention_mask)

        outputs = outputs.squeeze(-1) if outputs.shape[-1] == 1 else outputs

        loss = criterion(outputs, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(regression_model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

print("Training complete!")

"""## --- Step 5: Evaluation ---"""

# TODO: your code goes here

regression_model.eval()

with torch.no_grad():
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in tqdm(test_dataloader, desc="Evaluating"):

        input_ids_batch, attention_mask_batch, labels_batch = batch

        input_ids = input_ids_batch.to(device)
        attention_mask = attention_mask_batch.to(device)
        labels = labels_batch.to(device)

        optimizer.zero_grad()

        outputs = regression_model(input_ids=input_ids, attention_mask=attention_mask)

        outputs = outputs.squeeze(-1)

        loss = criterion(outputs, labels)
        running_loss += loss.item()

        all_preds.extend(outputs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(test_dataloader)
    print(f"Test Loss: {avg_loss:.4f}")

    from sklearn.metrics import mean_absolute_error, r2_score
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)

    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R^2 Score: {r2:.4f}")

"""# 2.Add Unsupervised Finetuning
In this step, you will perform unsupervised fine-tuning on the training dataset. This means the model will leverage only the SMILES strings without any corresponding labels to adapt its understanding of the data distribution. By familiarizing the model with the patterns and structure of the SMILES strings, you can potentially enhance its performance on downstream supervised tasks.

For this fine-tuning, you will use the Masked Language Modeling (MLM) objective, where the model learns to predict randomly masked tokens within the input sequence. Remember to save the fine-tuned model for later use.

"""

# TODO: your code goes here
tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
mlm_model = AutoModelForMaskedLM.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(mlm_model.parameters(), lr=5e-5, weight_decay=1e-2)

def mask_tokens(inputs, tokenizer, mask_prob=0.15, device="cuda"):
    """
    Function to randomly mask tokens for MLM.
    """
    inputs = inputs.to(device)
    rand = torch.rand(inputs.shape, device=device)


    mask_arr = (
        (rand < mask_prob)
        & (inputs != tokenizer.pad_token_id)
        & (inputs != tokenizer.cls_token_id)
        & (inputs != tokenizer.sep_token_id)
    )

    inputs[mask_arr] = tokenizer.mask_token_id

    return inputs, mask_arr.long()


EPOCHS = 3

for epoch in range(EPOCHS):
    mlm_model.train()
    running_loss = 0.0

    for batch in tqdm(train_dataloader, desc=f"MLM Fine-tuning Epoch {epoch+1}/{EPOCHS}"):
        input_ids_batch, attention_mask_batch, labels_batch = batch

        input_ids = input_ids_batch.to(device)
        attention_mask = attention_mask_batch.to(device)
        labels = labels_batch.to(device)

        optimizer.zero_grad()

        masked_inputs, labels = mask_tokens(input_ids.to(device), tokenizer, device=device)


        optimizer.zero_grad()

        outputs = mlm_model(input_ids=masked_inputs, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}/{EPOCHS} - MLM Loss: {avg_loss:.4f}")

mlm_model.save_pretrained("mlm_finetuned_model")
tokenizer.save_pretrained("mlm_finetuned_model")

print("MLM Fine-tuning complete! Model saved.")

"""# 3.Fine-Tune for Comparison
After performing unsupervised fine-tuning on the training data, we now fine-tune the model on the regression task with the regression head. By comparing the performance of the model before and after unsupervised fine-tuning, you can evaluate how the unsupervised fine-tuning impacts the model's performance on our target task.

"""

mlm_finetuned_model = AutoModel.from_pretrained("mlm_finetuned_model", trust_remote_code=True).to(device)

class RegressionHead(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(RegressionHead, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)

class MoLFormerWithRegression(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.regression_head = RegressionHead(input_dim=base_model.config.hidden_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        return self.regression_head(pooled_output).squeeze(-1)

regression_model = MoLFormerWithRegression(mlm_finetuned_model).to(device)

criterion = nn.MSELoss()
optimizer = optim.AdamW(regression_model.parameters(), lr=3e-5, weight_decay=1e-2)

EPOCHS = 5

for epoch in range(EPOCHS):
    regression_model.train()
    running_loss = 0.0

    for batch in tqdm(train_dataloader, desc=f"Regression Fine-tuning Epoch {epoch+1}/{EPOCHS}"):
        input_ids_batch, attention_mask_batch, labels_batch = batch

        input_ids = input_ids_batch.to(device)
        attention_mask = attention_mask_batch.to(device)
        labels = labels_batch.to(device)

        optimizer.zero_grad()

        outputs = regression_model(input_ids=input_ids, attention_mask=attention_mask)

        loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(regression_model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

# Save the fine-tuned regression model
# torch.save(regression_model.state_dict(), "/home/neuronet_team146/Project_Files/scripts/regression_finetuned_model.pth")
# torch.save(regression_model, "/home/neuronet_team146/Project_Files/scripts/regression_finetuned_model_full.pth")
# regression_model.save_pretrained("regression_finetuned_model")
# tokenizer.save_pretrained("mlm_finetuned_model")

# Save model state dictionary
# torch.save(regression_model.state_dict(), "/home/neuronet_team146/Project_Files/scripts/regression_model/regression_finetuned_model.pth")

# # Save the base model config so it can be reloaded
# regression_model.base_model.config.to_json_file("/home/neuronet_team146/Project_Files/scripts/regression_model/config.json")

