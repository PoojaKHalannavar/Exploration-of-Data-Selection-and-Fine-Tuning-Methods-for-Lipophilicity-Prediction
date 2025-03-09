import datasets
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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

print("MLM Model loaded successfully!")

# Dataset setup
DATASET_PATH = "scikit-fingerprints/MoleculeNet_Lipophilicity"
dataset = datasets.load_dataset(DATASET_PATH)
dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)

# Load filtered external dataset
filtered_external_data = pd.read_csv('/home/neuronet_team146/Project_Files/scripts/Cosine_Similarity_Filtered_Dataset.csv')

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

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Train Dataloader Size: {len(train_dataloader.dataset)}")
print(f"Test Dataloader Size: {len(test_dataloader.dataset)}")

# IA3 Adapter class
class IA3(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lambda_param = nn.Parameter(torch.ones(dim))  # Learnable scaling factor

    def forward(self, x):
        return x * self.lambda_param  # Element-wise scaling

# IA3 Wrapper for Attention Layers
class IA3Wrapper(nn.Module):
    def __init__(self, original_layer):
        super().__init__()
        self.original_layer = original_layer  
        self.ia3 = IA3(original_layer.in_features)  # IA3 scaling on input activations

    def forward(self, x):
        return self.original_layer(self.ia3(x))  # Apply IA3 before linear transformation

def modify_model(model):
    for name, module in model.named_modules():
        if any(x in name for x in ["query", "value"]):  
            if isinstance(module, nn.Linear):
                wrapped_layer = IA3Wrapper(module).to(module.weight.device)  # Wrap the layer
                parent_name = ".".join(name.split(".")[:-1])  # Get the parent module name
                layer_name = name.split(".")[-1]  # Get the layer name

                parent_module = model.get_submodule(parent_name)  # Get the parent module
                setattr(parent_module, layer_name, wrapped_layer)  # Replace layer with wrapped IA3 layer

                # Freeze original weights
                module.weight.requires_grad = False
                if module.bias is not None:
                    module.bias.requires_grad = False
    return model

# MoLFormer model with regression head
class MoLFormerWithRegressionHead(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.molformer = AutoModel.from_pretrained(model_name, trust_remote_code=True)
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

# Load and modify model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MoLFormerWithRegressionHead(MODEL_NAME).to(device)
model = modify_model(model).to(device)

# Print trainable parameters
print("--------------- Checking IA3 Layers -----------------------")
# for name, param in model.named_parameters():
    # if param.requires_grad:
#        print(name, param.shape)

print("==== Model with IA3 Adaptation Loaded Successfully! ===================")
#print(model)

# Optimizer and loss function
# optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-6, weight_decay=0.01)

criterion = nn.MSELoss()

# Training loop
print("-------------Train started!---------------")
model.train()
EPOCHS = 10

for epoch in range(EPOCHS):
    running_loss = 0.0
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        input_ids_batch, attention_mask_batch, labels_batch = batch
        input_ids = input_ids_batch.to(device)
        attention_mask = attention_mask_batch.to(device)
        labels = labels_batch.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        outputs = outputs.squeeze(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

print("Training complete!")

# Evaluation
model.eval()
with torch.no_grad():
    running_loss = 0.0
    all_preds, all_labels = [], []
    for batch in tqdm(test_dataloader, desc="Evaluating"):
        input_ids_batch, attention_mask_batch, labels_batch = batch
        input_ids = input_ids_batch.to(device)
        attention_mask = attention_mask_batch.to(device)
        labels = labels_batch.to(device)
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
