import os
import pandas as pd
import numpy as np

def get_dataset_dir(dataset_name: str) -> str:
    """
    Get the appropriate directory path for a dataset depending on the environment.
    If running on Kaggle, the path is prefixed with /kaggle/input/.
    Otherwise, it looks for the dataset in a designated local directory.
    
    Args:
        dataset_name (str): The name of the dataset folder (e.g., 'stanford-rna-3d-folding-part-2').
        
    Returns:
        str: Absolute path to the dataset.
    """
    is_kaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '') != ''
    
    if is_kaggle:
        return os.path.join('/kaggle/input', dataset_name)
    else:
        # Defaults to a local directory (adjust according to your local setup)
        return os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'data', dataset_name)

def load_competition_data(csv_path: str, is_train: bool = False) -> pd.DataFrame:
    """
    Load the Kaggle competition target data.
    
    Args:
        csv_path (str): Path to the train.csv or test.csv file.
        is_train (bool): If True, might perform additional filtering or processing required for train set.
        
    Returns:
        pd.DataFrame: DataFrame containing target IDs and sequences.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Competition CSV file not found at {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    # Optionally handled additional fields specific to the train set (e.g., temporal cutoff filtering here)
    if is_train:
        # For example, filter by experimental date if available in the training metadata
        pass
        
    return df

def load_msa(a3m_path: str) -> str:
    """
    Load Multiple Sequence Alignment from an .a3m file.
    
    Args:
        a3m_path (str): The path to the .a3m file.
        
    Returns:
        str: The raw content of the .a3m file.
    """
    if not os.path.exists(a3m_path):
        return "" # Or raise an error, depending on fallback strategies
        
    with open(a3m_path, 'r') as f:
        msa_content = f.read()
    return msa_content

def load_foundation_embeddings(target_id: str, emb_dir: str) -> np.ndarray:
    """
    Load pre-computed foundation model embeddings (e.g., AIDO.RNA or RNA-FM) for a given target.
    
    Args:
        target_id (str): The ID of the RNA sequence.
        emb_dir (str): Directory where the pre-computed embeddings are stored.
        
    Returns:
        np.ndarray: The embedding tensor as a numpy array.
    """
    # Look for .npy or .npz format embeddings
    emb_path_npy = os.path.join(emb_dir, f"{target_id}.npy")
    emb_path_npz = os.path.join(emb_dir, f"{target_id}.npz")
    
    if os.path.exists(emb_path_npy):
        return np.load(emb_path_npy)
    elif os.path.exists(emb_path_npz):
        data = np.load(emb_path_npz)
        # Assuming the primary embedding is stored under a key like 'embedding' or 'arr_0'
        keys = list(data.keys())
        if 'embedding' in keys:
            return data['embedding']
        return data[keys[0]]
    else:
        raise FileNotFoundError(f"No embeddings found for target {target_id} in {emb_dir}")
