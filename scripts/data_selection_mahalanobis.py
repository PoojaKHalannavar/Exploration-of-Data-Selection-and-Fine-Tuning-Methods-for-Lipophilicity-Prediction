import numpy as np
from scipy.spatial import distance
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics.pairwise import cosine_similarity
import datasets


def smiles_to_fingerprint_vector(smiles, radius=2, n_bits=2048):
    # Convert to a numpy array (binary vector) from an RDKit fingerprint
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((1,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    return None

def compute_mahalanobis_distance(x, mean, cov_inv):
    return distance.mahalanobis(x, mean, cov_inv)

'''
Mahalanobis distance is a measure that accounts for the correlations among features by considering the covariance structure of the target data. 
Compute the mean and covariance of the target fingerprints.
Calculate the Mahalanobis distance for each external fingerprint.
Choose external samples whose distance from the target distribution is below a defined threshold.

How it works - 
Convert SMILES to fingerprint vectors (as numpy arrays).
Compute the target distributions' mean and covariance.
Calculate the distance for each external sample, then filter based on a threshold (which can be set, for instance, according to a chi-square distribution quantile).
'''
def filter_external_data_mahalanobis(target_smiles, external_data, distance_threshold=3.0):
    # Convert SMILES to fingerprint vectors for target and external datasets
    target_fps = [smiles_to_fingerprint_vector(smiles) for smiles in target_smiles]
    external_fps = [smiles_to_fingerprint_vector(smiles) for smiles in external_data['SMILES']]

    # Remove None values
    target_fps = [fp for fp in target_fps if fp is not None]
    external_fps = [fp for fp in external_fps if fp is not None]

    # Stack arrays to form 2D arrays
    target_arr = np.array(target_fps)
    external_arr = np.array(external_fps)

    # Compute the mean and covariance matrix of the target data
    mean_target = np.mean(target_arr, axis=0)
    cov_target = np.cov(target_arr, rowvar=False)
    
    # Regularize covariance matrix to avoid singularity issues
    cov_target += np.eye(cov_target.shape[0]) * 1e-6
    cov_inv = np.linalg.inv(cov_target)

    selected_indices = []
    for idx, ext in enumerate(external_arr):
        d = compute_mahalanobis_distance(ext, mean_target, cov_inv)
        if d <= distance_threshold:
            selected_indices.append(idx)
    
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
filtered_external_data = filter_external_data_mahalanobis(target_smiles, external_data)
print(f"====================== Mahalanobis filtered external data length : - ", len(filtered_external_data))

# Save filtered external data to a new CSV file
filtered_external_data.to_csv('/home/neuronet_team146/Project_Files/scripts/Mahalanobis_Similarity_Filtered_Dataset.csv', index=False)

print(f"Filtered external data saved to '/home/neuronet_team146/Project_Files/scripts/Mahalanobis_Similarity_Filtered_Dataset.csv'")
