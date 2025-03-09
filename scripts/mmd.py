from sklearn.metrics import pairwise_kernels
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))
    return None

def compute_mmd(X, Y, kernel='rbf', gamma=0.1):
    """
    Compute Maximum Mean Discrepancy between X and Y using kernel trick.
    """
    K_XX = pairwise_kernels(X, X, metric=kernel, gamma=gamma)
    K_YY = pairwise_kernels(Y, Y, metric=kernel, gamma=gamma)
    K_XY = pairwise_kernels(X, Y, metric=kernel, gamma=gamma)
    
    return np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)

def select_mmd_data(target_smiles, external_data, top_k=100):
    target_fps = [smiles_to_fingerprint(smiles) for smiles in target_smiles]
    target_fps = np.array([fp for fp in target_fps if fp is not None])

    external_fps = [smiles_to_fingerprint(smiles) for smiles in external_data['SMILES']]
    valid_external = external_data[[fp is not None for fp in external_fps]]
    external_fps = np.array([fp for fp in external_fps if fp is not None])

    if len(target_fps) == 0 or len(external_fps) == 0:
        raise ValueError("No valid fingerprints found.")

    # Calculate MMD distance for each external sample
    mmd_scores = []
    for i in tqdm(range(len(external_fps))):
        mmd = compute_mmd(target_fps, [external_fps[i]])
        mmd_scores.append(mmd)

    valid_external['MMD_Score'] = mmd_scores
    selected_data = valid_external.nsmallest(top_k, 'MMD_Score')
    
    return selected_data

# Load datasets
dataset = datasets.load_dataset("scikit-fingerprints/MoleculeNet_Lipophilicity")['train'].train_test_split(test_size=0.2, seed=42)
external_data = pd.read_csv('/home/neuronet_team146/Project_Files/scripts/External_Dataset_for_Task2.csv')

# Select External Data using MMD
target_smiles = dataset['train']['SMILES']
filtered_data = select_mmd_data(target_smiles, external_data)

# Save filtered dataset
filtered_data.to_csv('/home/neuronet_team146/Project_Files/scripts/MMD_Filtered_Dataset.csv', index=False)
print(f"Filtered external data (MMD) saved to '/home/neuronet_team146/Project_Files/scripts/MMD_Filtered_Dataset.csv'")
