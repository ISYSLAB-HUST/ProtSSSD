""" 
"""
from torch.utils.data import DataLoader,Dataset, random_split
from torch.nn.utils.rnn import pad_sequence
import json
import numpy as np
import torch

def dihedral_to_radians(angle: torch.tensor) -> torch.tensor:
    """ Converts angles to radians
    Args:
        angles: tensor containing angle values
    """
    radians = angle * np.pi / 180
    radians = torch.stack((torch.sin(radians), torch.cos(radians)),dim=-1)
    return radians

class JSONDataset(Dataset):
    def __init__(self, dir_name,) -> None:
        super().__init__()
        self.data = json.load(open(dir_name))

    def __getitem__(self, index):
        record  = self.data[index]
        return record
    
    def __len__(self):
        return len(self.data)
    
def collate_fn(batch_record):
    ss8_batch = pad_sequence([torch.tensor(item["ss8"]) for item in batch_record], batch_first=True)
    ss3_batch = pad_sequence([torch.tensor(item["ss3"]) for item in batch_record], batch_first=True)
    tokens_batch = [(item["id"], item["primary"]) for item in batch_record]
    disorder_batch = pad_sequence([torch.tensor(item["disorder"], dtype=torch.int) for item in batch_record], batch_first=True)
    valid_mask_batch = pad_sequence([torch.tensor(item["valid_mask"], dtype=torch.int) for item in batch_record], batch_first=True)

    psi_batch = pad_sequence([dihedral_to_radians(torch.tensor(item["psi"])) for item in batch_record], batch_first=True)
    phi_batch = pad_sequence([dihedral_to_radians(torch.tensor(item["phi"])) for item in batch_record], batch_first=True)
    rsa_batch = pad_sequence([torch.tensor(item["rsa"]) for item in batch_record], batch_first=True)

    return {"ss8":ss8_batch, "ss3":ss3_batch, "seq":tokens_batch, "disorder":disorder_batch, "valid_mask":valid_mask_batch,
            "psi":psi_batch, "phi":phi_batch, "rsa":rsa_batch}

def casp_collate_fn(batch_record):
    ss8_batch = pad_sequence([torch.tensor(item["ss8"]) for item in batch_record], batch_first=True)
    ss3_batch = pad_sequence([torch.tensor(item["ss3"]) for item in batch_record], batch_first=True)
    tokens_batch = [(item["id"], item["primary"]) for item in batch_record]
    
    valid_mask_batch = pad_sequence([torch.ones(len(item["primary"]), dtype=torch.int) for item in batch_record], batch_first=True)

    psi_batch = pad_sequence([dihedral_to_radians(torch.tensor(item["psi"])) for item in batch_record], batch_first=True)
    phi_batch = pad_sequence([dihedral_to_radians(torch.tensor(item["phi"])) for item in batch_record], batch_first=True)
    rsa_batch = pad_sequence([torch.tensor(item["rsa"]) for item in batch_record], batch_first=True)

    return {"ss8":ss8_batch, "ss3":ss3_batch, "seq":tokens_batch, "valid_mask":valid_mask_batch,
            "psi":psi_batch, "phi":phi_batch, "rsa":rsa_batch}


if __name__ == "__main__":
    test_dataset = JSONDataset("/mnt/d/netsurfp/NetSurfP-3.0/dataset/secondary_structure/secondary_structure/secondary_structure_ts115.json")
    loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)
    for item in loader:
        print(item['seq'][0][0], item['valid_mask'].sum())


