# This file fine-tunes the model using LORA method and gives MAE and R2 Score for comparison

import datasets
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the fine-tuned MLM model and tokenizer
MODEL_NAME = "mlm_finetuned_model"
mlm_model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Dataset setup
DATASET_PATH = "scikit-fingerprints/MoleculeNet_Lipophilicity"
dataset = datasets.load_dataset(DATASET_PATH)
dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)


# Load filtered external dataset
filtered_external_data = pd.read_csv('/home/neuronet_team146/Project_Files/scripts/task3_data_selection/Cosine_Similarity_Filtered_Dataset.csv')

# Standardize column names
train_data_df = dataset['train'].to_pandas()
filtered_external_data.rename(columns={'Label': 'label'}, inplace=True)

# Combine datasets
combined_data = pd.concat([train_data_df[['SMILES', 'label']], filtered_external_data[['SMILES', 'label']]], ignore_index=True)

print(f"Original Train Data: {len(train_data_df)}")
print(f"Filtered External Data: {len(filtered_external_data)}")
print(f"Combined Data: {len(combined_data)}")

# Tokenization function
def tokenize_function(smiles_list):
    return tokenizer(smiles_list, truncation=True, padding="max_length", max_length=128, return_tensors="pt")

# Tokenize combined training data
tokenized_combined_data = tokenize_function(combined_data['SMILES'].tolist())

# Convert to PyTorch tensors
def convert_to_torch(tokenized_data, labels):
    input_ids = tokenized_data['input_ids']
    attention_mask = tokenized_data['attention_mask']
    labels = torch.tensor(labels, dtype=torch.float)
    return TensorDataset(input_ids, attention_mask, labels)

# Convert combined data to tensor dataset
train_dataset = convert_to_torch(tokenized_combined_data, combined_data['label'].tolist())

# Tokenize and convert test data
tokenized_test_data = tokenize_function(dataset['test']['SMILES'])
test_labels = dataset['test']['label']
test_dataset = convert_to_torch(tokenized_test_data, test_labels)

# Split training dataset into training and validation sets (80% train, 20% validation)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

# Create DataLoaders
train_dataloader = DataLoader(train_subset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_subset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Train Dataloader Size: {len(train_dataloader.dataset)}")
print(f"Validation Dataloader Size: {len(val_dataloader.dataset)}")
print(f"Test Dataloader Size: {len(test_dataloader.dataset)}")

# LoRA Adapter
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank=16, alpha=32):
        super().__init__()
        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(rank, out_features) * 0.01)

    def forward(self, x):
        return x @ self.A @ self.B * self.scaling

def modify_model(model):
    for name, module in model.named_modules():
        if any(x in name for x in ["query", "value"]):
            if isinstance(module, nn.Linear):
                in_dim, out_dim = module.weight.shape
                lora_adapter = LoRA(in_dim, out_dim).to(module.weight.device)

                original_forward = module.forward

                def new_forward(x):
                    return original_forward(x) + lora_adapter(x)

                module.forward = new_forward
                module.weight.requires_grad = False
                if module.bias is not None:
                    module.bias.requires_grad = False

    return model

# MLM Model with Regression Head
class MLMWithRegressionHead(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.mlm = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.regression_head = nn.Sequential(
            nn.Linear(self.mlm.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.mlm(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        return self.regression_head(pooled_output).squeeze(-1)

# Load model and apply LoRA
model = MLMWithRegressionHead(MODEL_NAME).to(device)
model = modify_model(model)

print("--------------- Checking LORA Layers -----------------------")
# for name, param in model.named_parameters():
    # if param.requires_grad:
        #print(name, param.shape)

print("==== Model with Regression Head Loaded Successfully! ===================")

# Optimizer & Loss
optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
criterion = nn.MSELoss()

# Training Loop
print("-------------Train started!---------------")
model.train()
EPOCHS = 15

for epoch in range(EPOCHS):
    running_loss = 0.0
    val_running_loss = 0.0
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        input_ids, attention_mask, labels = (b.to(device) for b in batch)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_dataloader)

    # Validation Loss
    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc=f"Validation Epoch {epoch+1}/{EPOCHS}"):
            input_ids, attention_mask, labels = (b.to(device) for b in batch)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            loss = criterion(outputs, labels)
            val_running_loss += loss.item()

    avg_val_loss = val_running_loss / len(val_dataloader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

    # Reset model to training mode
    model.train()

print("Training complete!")

# Evaluation
model.eval()
with torch.no_grad():
    running_loss = 0.0
    all_preds, all_labels = [], []

    for batch in tqdm(test_dataloader, desc="Evaluating"):
        input_ids, attention_mask, labels = (b.to(device) for b in batch)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

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

