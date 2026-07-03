import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob

from model import QuoridorNet

class RealQuoridorDataset(Dataset):
    def __init__(self, data_dir="../data", max_history=3):
        self.data = []
        
        # Find all iteration directories
        iter_dirs = glob.glob(os.path.join(data_dir, "iter_*"))
        iter_dirs.sort()
        
        if len(iter_dirs) > max_history:
            iter_dirs = iter_dirs[-max_history:]
            
        pt_files = []
        for d in iter_dirs:
            pt_files.extend(glob.glob(os.path.join(d, "*.pt")))
            
        print(f"Loading data from {len(iter_dirs)} iterations: {[os.path.basename(d) for d in iter_dirs]}")
        
        for file in pt_files:
            try:
                game_data = torch.load(file)
                self.data.extend(game_data)
            except Exception as e:
                print(f"Error loading {file}: {e}")
                
        print(f"Loaded {len(pt_files)} games, totaling {len(self.data)} steps.")
        
    def __len__(self):
        return max(1, len(self.data))
        
    def __getitem__(self, idx):
        if len(self.data) == 0:
            # Fallback if no data is found to prevent crashing
            return (torch.randn(9, 17, 17), torch.ones(132, dtype=torch.bool),
                    torch.zeros(132), torch.zeros(1), torch.zeros(1), torch.zeros(1, 17, 17))
            
        step = self.data[idx]
        return (step['state'], step['mask'], step['target_policy'], 
                step['target_value'], step['target_distance'], step['target_bottleneck'])

def quoridor_loss(policy_logits, value, distance, bottleneck, 
                  target_policy, target_value, target_distance, target_bottleneck):
    policy_loss = F.cross_entropy(policy_logits, target_policy)
    value_loss = nn.MSELoss()(value, target_value)
    distance_loss = nn.MSELoss()(distance, target_distance)
    bottleneck_loss = nn.BCELoss()(bottleneck, target_bottleneck)
    
    total_loss = policy_loss + 1.0 * value_loss + 0.05 * distance_loss + 0.05 * bottleneck_loss
    return total_loss, policy_loss, value_loss, distance_loss, bottleneck_loss

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = QuoridorNet().to(device)
    
    model_path = "../models/best_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Loaded existing model weights.")
        
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    dataset = RealQuoridorDataset(data_dir="../data")
    if len(dataset.data) == 0:
        print("No training data found. Skipping training.")
        return
        
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    epochs = 3
    print("Starting Training Loop on Real Data...")
    for epoch in range(epochs):
        model.train()
        total_epoch_loss = 0.0
        
        for batch_idx, (state, mask, t_pol, t_val, t_dist, t_bot) in enumerate(dataloader):
            state = state.to(device)
            mask = mask.to(device)
            t_pol = t_pol.to(device)
            t_val = t_val.to(device)
            t_dist = t_dist.to(device)
            t_bot = t_bot.to(device)
            
            optimizer.zero_grad()
            
            p_logits, v, d, b = model(state, mask)
            loss, p_loss, v_loss, d_loss, b_loss = quoridor_loss(p_logits, v, d, b, t_pol, t_val, t_dist, t_bot)
            
            loss.backward()
            optimizer.step()
            
            total_epoch_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] - Total Loss: {total_epoch_loss/len(dataloader):.4f}")
        
    # Save the updated model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Saved updated model to {model_path}")

if __name__ == "__main__":
    train()
