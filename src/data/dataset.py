import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random

class MRIT1T2Dataset(Dataset):
    def __init__(self, t1_dir, t2_dir, mode='train', transform=None, slice_mode='middle', paired=False):
        """
        Args:
            t1_dir (str): Directory with T1 MRI scans
            t2_dir (str): Directory with T2 MRI scans
            mode (str): 'train', 'val', or 'test'
            transform: Optional transforms to be applied
            slice_mode (str): 'middle' or 'random' for slice selection
            paired (bool): Whether to use paired or unpaired data
        """
        self.t1_dir = t1_dir
        self.t2_dir = t2_dir
        self.transform = transform
        self.slice_mode = slice_mode
        self.paired = paired
        
        # Get list of files
        self.t1_files = sorted([f for f in os.listdir(t1_dir) if f.endswith('.nii.gz')])
        self.t2_files = sorted([f for f in os.listdir(t2_dir) if f.endswith('.nii.gz')])
        
        # Split data based on mode
        if mode == 'train':
            self.t1_files = self.t1_files[:int(0.7*len(self.t1_files))]
            self.t2_files = self.t2_files[:int(0.7*len(self.t2_files))]
        elif mode == 'val':
            self.t1_files = self.t1_files[int(0.7*len(self.t1_files)):int(0.85*len(self.t1_files))]
            self.t2_files = self.t2_files[int(0.7*len(self.t2_files)):int(0.85*len(self.t2_files))]
        else:  # test
            self.t1_files = self.t1_files[int(0.85*len(self.t1_files)):]
            self.t2_files = self.t2_files[int(0.85*len(self.t2_files)):]

    def __len__(self):
        return len(self.t1_files)

    def normalize_volume(self, volume):
        """Normalize volume to [0,1] range"""
        min_val = np.min(volume)
        max_val = np.max(volume)
        return (volume - min_val) / (max_val - min_val + 1e-8)

    def get_slice(self, volume):
        """Extract slice based on slice_mode"""
        if self.slice_mode == 'middle':
            slice_idx = volume.shape[2] // 2
        else:  # random
            slice_idx = random.randint(0, volume.shape[2] - 1)
        return volume[:, :, slice_idx]

    def __getitem__(self, idx):
        # Load T1 volume
        t1_path = os.path.join(self.t1_dir, self.t1_files[idx])
        t1_volume = nib.load(t1_path).get_fdata()
        t1_volume = self.normalize_volume(t1_volume)
        t1_slice = self.get_slice(t1_volume)
        
        # Load T2 volume (paired or random)
        if self.paired:
            t2_idx = idx
        else:
            t2_idx = random.randint(0, len(self.t2_files) - 1)
            
        t2_path = os.path.join(self.t2_dir, self.t2_files[t2_idx])
        t2_volume = nib.load(t2_path).get_fdata()
        t2_volume = self.normalize_volume(t2_volume)
        t2_slice = self.get_slice(t2_volume)
        
        # Convert to tensors
        t1_tensor = torch.from_numpy(t1_slice).float().unsqueeze(0)
        t2_tensor = torch.from_numpy(t2_slice).float().unsqueeze(0)
        
        # Apply transforms if any
        if self.transform:
            t1_tensor = self.transform(t1_tensor)
            t2_tensor = self.transform(t2_tensor)
            
        return {'T1': t1_tensor, 'T2': t2_tensor}

def get_transforms():
    """Return standard transforms for MRI data"""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

def create_dataloaders(t1_dir, t2_dir, batch_size=1, num_workers=4, paired=False):
    """Create train, validation and test dataloaders"""
    transform = get_transforms()
    
    # Create datasets
    train_dataset = MRIT1T2Dataset(t1_dir, t2_dir, mode='train', 
                                 transform=transform, paired=paired)
    val_dataset = MRIT1T2Dataset(t1_dir, t2_dir, mode='val', 
                               transform=transform, paired=paired)
    test_dataset = MRIT1T2Dataset(t1_dir, t2_dir, mode='test', 
                                transform=transform, paired=paired)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test the dataset
    t1_dir = "data/IXI_T1"
    t2_dir = "data/IXI_T2"
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(t1_dir, t2_dir)
    
    # Test a batch
    batch = next(iter(train_loader))
    print("Batch shapes:")
    print(f"T1: {batch['T1'].shape}")
    print(f"T2: {batch['T2'].shape}")
    print("\nValue ranges:")
    print(f"T1: [{batch['T1'].min():.3f}, {batch['T1'].max():.3f}]")
    print(f"T2: [{batch['T2'].min():.3f}, {batch['T2'].max():.3f}]") 