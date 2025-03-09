import gc
import torch
import torch.nn as nn
from torch.autograd import grad
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig
from Task1 import MoLFormerWithRegression, test_dataloader
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
gc.collect()

# Load the fine-tuned model and tokenizer
mlm_model = AutoModel.from_pretrained("mlm_finetuned_model", trust_remote_code=True).to(device)
regression_model = MoLFormerWithRegression(mlm_model).to(device)
tokenizer = AutoTokenizer.from_pretrained("mlm_finetuned_model", trust_remote_code=True)

print("Model loaded successfully!")

# Load the external dataset
external_data = pd.read_csv("/home/neuronet_team146/Project_Files/scripts/External_Dataset_for_Task2.csv")

# Tokenize and create DataLoader
def tokenize_data(smiles_list, tokenizer):
    encoded = tokenizer(smiles_list, padding=True, truncation=True, return_tensors="pt")
    return encoded["input_ids"], encoded["attention_mask"]

# Tokenize SMILES strings
external_input_ids, external_attention_mask = tokenize_data(external_data["SMILES"].tolist(), tokenizer)

# Convert labels
external_labels = torch.tensor(external_data["Label"].values, dtype=torch.float16)

# Create DataLoader for external dataset
external_dataset = TensorDataset(external_input_ids, external_attention_mask, external_labels)
external_dataloader = DataLoader(external_dataset, batch_size=4, shuffle=False, pin_memory=True)

# Compute gradients of loss w.r.t. model parameters
def compute_gradients(model, dataloader):
    model.eval() # changed
    grads = []
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler()
    for batch in tqdm(dataloader, desc="Computing Gradients"):
        
        input_ids_batch, attention_mask_batch, labels_batch = batch
        input_ids = input_ids_batch.to(device)
        attention_mask = attention_mask_batch.to(device)
        labels = labels_batch.to(device)

        # model.zero_grad()
        # outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # outputs = outputs.squeeze(-1)
        
        # loss = criterion(outputs, labels)

        # loss_grads = grad(loss, model.parameters(), retain_graph=False)
        # grads.append([g.detach().to(device) for g in loss_grads])

        # Mixed precision training
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            outputs = outputs.squeeze(-1)

            loss = criterion(outputs, labels)

        # Compute gradients
        loss_grads = grad(loss, model.parameters(), retain_graph=False)

        # Move gradients to CPU immediately to free up GPU memory
        grads.append([g.detach().cpu() for g in loss_grads])


        # Free memory
        del input_ids, attention_mask, labels, outputs, loss, loss_grads
        torch.cuda.empty_cache()
        gc.collect()

    return grads  # List of gradients for each batch

# Compute gradients for the test set
test_grads = compute_gradients(regression_model, test_dataloader)

# Compute gradients for the external dataset
external_grads = compute_gradients(regression_model, external_dataloader)

# Implement LiSSA for Hessian-Vector Product (HVP) Approximation
def lissa_inverse_hvp2(model, test_grads, damping=0.01, scale=10, num_iter=500):
    ihvp = [torch.zeros_like(p, device=device) for p in model.parameters()]

    for i, test_grad in enumerate(test_grads):
        v = test_grad
        cur_estimate = test_grad

        for j in range(num_iter):
            model.zero_grad()

            # Compute Hessian-vector product
            hvp = grad(
                sum((g * e).sum() for g, e in zip(test_grad, cur_estimate)),
                model.parameters(),
                retain_graph=True
            )

            hvp = [h.detach().to(device) for h in hvp]

            # Update estimate using damping and scaling
            cur_estimate = [v_i + (1 - damping) * cur_e_i - h_i / scale for v_i, cur_e_i, h_i in zip(v, cur_estimate, hvp)]

        ihvp = [i_hvp + cur_e_i for i_hvp, cur_e_i in zip(ihvp, cur_estimate)]

    return [i_hvp / len(test_grads) for i_hvp in ihvp]  # Normalize


def lissa_inverse_hvp(model, test_grads, damping=0.01, scale=10, num_iter=100):
    # Ensure ihvp is on the same device
    ihvp = [torch.zeros_like(p, device=device) for p in model.parameters()]

    for test_grad in test_grads:
        v = [g.clone().to(device).float().requires_grad_(True) for g in test_grad]  # Ensure test_grad is on device
        cur_estimate = [g.clone().to(device).float().requires_grad_(True) for g in test_grad]  # Move to device

        for _ in range(num_iter):
            model.zero_grad()

            with torch.cuda.amp.autocast():  # Enable mixed precision for memory saving
                # Compute Hessian-vector product
                outputs = sum((g.to(device) * e.to(device)).sum() for g, e in zip(test_grad, cur_estimate))  # Ensure tensors are on device
            hvp = grad(
                outputs=outputs,
                inputs=model.parameters(),
                retain_graph=True,
                create_graph=True,
                allow_unused=True
            )

            # Ensure hvp is on device and handle None values
            hvp = [
                h.to(device) if h is not None else torch.zeros_like(e).to(device)
                for h, e in zip(hvp, cur_estimate)
            ]

            # Update LiSSA estimate using damping and scaling
            cur_estimate = [
                v_i + (1 - damping) * cur_e_i - h_i / scale
                for v_i, cur_e_i, h_i in zip(v, cur_estimate, hvp)
            ]

        # Ensure ihvp stays on the correct device
        ihvp = [
            i_hvp.to(device) + cur_e_i.to(device)
            for i_hvp, cur_e_i in zip(ihvp, cur_estimate)
        ]

    # Normalize across test gradients
    return [i_hvp / len(test_grads) for i_hvp in ihvp]


# Compute inverse Hessian-Vector Product
inverse_hvp = lissa_inverse_hvp(regression_model, test_grads)

# Compute Influence Scores
def compute_influence_scores(inverse_hvp, external_grads):
    influence_scores = []

    for ext_grad in external_grads:
        influence = sum((ihvp * ext_g).sum().item() for ihvp, ext_g in zip(inverse_hvp, ext_grad))
        influence_scores.append(influence)

    return influence_scores

# Get influence scores for external data points
influence_scores = compute_influence_scores(inverse_hvp, external_grads)

# Add influence scores to the external dataset
external_data["Influence"] = influence_scores

# Save ranked dataset
external_data_sorted = external_data.sort_values(by="Influence", ascending=False)
external_data_sorted.to_csv("External-Dataset_with_Influence.csv", index=False)

print("Influence scores computed and saved!")
