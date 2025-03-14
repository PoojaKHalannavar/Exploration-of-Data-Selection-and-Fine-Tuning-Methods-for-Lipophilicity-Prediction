# This file computes Influence scores for data points and save in a csv file

import torch
from torch.utils.data import Dataset, DataLoader
import sklearn
import datasets
import numpy as np
import transformers
import pandas as pd
from tqdm import tqdm
import random
from transformers import AdamW, AutoTokenizer, AutoModel, AutoConfig

DATASET_PATH = "scikit-fingerprints/MoleculeNet_Lipophilicity"
MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"

class MoLFormerWithRegressionHead(torch.nn.Module):
    def __init__(self, model_name="ibm/MoLFormer-XL-both-10pct"):
        super().__init__()
        self.molformer = transformers.AutoModel.from_pretrained(
            model_name, deterministic_eval=True, trust_remote_code=True
        )
        self.regression_head = torch.nn.Linear(self.molformer.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.molformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        regression_output = self.regression_head(pooled_output)

        return regression_output

class ExtSMILESDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)  
        self.smiles = self.data["SMILES"].tolist()
        self.labels = torch.tensor(self.data["Label"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.smiles[idx], self.labels[idx]

import gc
gc.collect()

import torch
torch.cuda.empty_cache()

def compute_influence_scores(model, train_dataloader, test_dataloader, tokenizer, device, gpu=-1, damping=0.01,
                       scale=25.0, recursion_depth=5000, r=1):
    model.eval()
    model.to(device)

    def compute_loss(predictions, targets):
        loss = torch.nn.MSELoss()
        return loss(predictions, targets.view(-1, 1))

    def compute_gradients(inputs, targets):
        if gpu >= 0:
            inputs, targets = inputs.to(device), targets.to(device)
        targets = targets.float()
        predictions = model(**inputs)
        gc.collect()

        loss = compute_loss(predictions, targets).float()
        params = []
        for param in model.parameters():
          if param.requires_grad:
            params.append(param)

        return list(torch.autograd.grad(loss, params, create_graph=True))

    def compute_hessian_vector_product(loss, params, vector):
        grad_params = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
        dot_product = sum(torch.sum(gradp * vec) for gradp, vec in zip(grad_params, vector))
        hvp = torch.autograd.grad(dot_product, params, retain_graph=True)
        return hvp

    def compute_influence_for_test_point(test_inputs, test_targets):
        test_influence_sum = None
        for _ in range(r):
            gradients_for_test = compute_gradients(test_inputs, test_targets)
            test_influence_estimate = [vi.clone() for vi in gradients_for_test]
            for _ in range(recursion_depth):
                for x_train, t_train in train_dataloader:
                    x_train = tokenizer([x_train[0]], return_tensors="pt")
                    if gpu >= 0:
                        x_train, t_train = x_train.to(device), t_train.to(device)
                    y_train = model(**x_train)
                    gc.collect()

                    loss_train = compute_loss(y_train, t_train)
                    params = [p for p in model.parameters() if p.requires_grad]
                    hv = compute_hessian_vector_product(loss_train, params, test_influence_estimate)
                    test_influence_estimate = [grad + (1 - damping) * est_grad - hvi / scale for grad, est_grad, hvi in zip(gradients_for_test, test_influence_estimate, hv)]
                    break
            if test_influence_sum is None:
                test_influence_sum = test_influence_estimate
            else:
                test_influence_sum = [a + b for a, b in zip(test_influence_sum, test_influence_estimate)]
        return [influence / r for influence in test_influence_sum]


    aggregated_influence_scores_for_test = None
    num_test_points = 0
    for batch in test_dataloader:
        gc.collect()
        model.eval()
        test_inputs, test_targets = batch["SMILES"], batch["label"]
        test_inputs = tokenizer(test_inputs, return_tensors="pt", padding=True)
        if test_inputs["input_ids"].dim() == 3:
            test_inputs["input_ids"] = test_inputs["input_ids"].unsqueeze(0)
        if test_targets.dim() == 0:
            test_targets = test_targets.unsqueeze(0)
        test_influence_for_point = compute_influence_for_test_point(test_inputs, test_targets)
        if aggregated_influence_scores_for_test is None:
            aggregated_influence_scores_for_test = test_influence_for_point
        else:
            aggregated_influence_scores_for_test = [a + b for a, b in zip(aggregated_influence_scores_for_test, test_influence_for_point)]
        num_test_points += 1
    if num_test_points > 0:
        aggregated_influence_scores_for_test = [s / num_test_points for s in aggregated_influence_scores_for_test]
    else:
        raise ValueError("Test loader is empty.")

    influence_scores = []
    total_train = len(train_dataloader.dataset)
    for x_train, t_train in train_dataloader:
        batch_size = 1
        for i in range(batch_size):
            train_input = tokenizer(x_train, return_tensors="pt", padding=True)
            target = t_train[i].unsqueeze(0)
            if train_input["input_ids"].dim() == 3:
              train_input["input_ids"] = train_input["input_ids"].unsqueeze(0)
            if target.dim() == 0:
              target = target.unsqueeze(0)
            grad_train = compute_gradients(train_input, target)
            dot_prod = sum(torch.sum(g * s).item() for g, s in zip(grad_train, aggregated_influence_scores_for_test))
            influence_scores.append(- dot_prod / total_train)
    return influence_scores


if __name__ == "__main__":
    # Define the full path to the model directory
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/home/neuronet_team146/Project_Files/scripts/mlm_finetuned_model"

    # Load the fine-tuned model and tokenizer
    mlm_model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)
    regression_model = MoLFormerWithRegressionHead(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model = regression_model
    model.train()

    model.to(device)

    loss = torch.nn.MSELoss()
    dataset = ExtSMILESDataset("/home/neuronet_team146/Project_Files/scripts/External_Dataset_for_Task2.csv")
    ext_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    main_dataset = datasets.load_dataset(DATASET_PATH)
    split_dataset = main_dataset["train"].train_test_split(
        test_size=0.05,
        shuffle=True,
        seed=42
    )
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    influences = compute_influence_scores(model, ext_dataloader, test_dataloader, tokenizer, device, gpu=1,
                                damping=0.01, scale=25.0, recursion_depth=5, r=1)


external_data = pd.read_csv("/home/neuronet_team152/Fine-tune-a-Chemical-Language-Model/scripts/External_Dataset_for_Task2.csv")
# print("length of influences: ",len(influences))
external_data["Influence"] = influences
external_data_sorted = external_data.sort_values(by="Influence", ascending=False)
external_data_sorted.to_csv("External-Dataset_with_Influence.csv", index=False)
