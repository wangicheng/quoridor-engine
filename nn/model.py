import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalPoolingResBlock(nn.Module):
    """
    KataGo-inspired Residual Block with Global Pooling.
    Extracts global context and injects it back into the spatial features,
    crucial for Quoridor where a wall placed anywhere can change the global shortest path.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # Global Pooling branch
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc1 = nn.Linear(channels, channels // 2)
        self.global_fc2 = nn.Linear(channels // 2, channels)

    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Global Pooling Context
        # B, C, H, W -> B, C, 1, 1 -> B, C
        g = self.global_pool(out).squeeze(-1).squeeze(-1) 
        g = F.relu(self.global_fc1(g))
        g = self.global_fc2(g)
        # Broadcast back to spatial dimensions
        g = g.unsqueeze(-1).unsqueeze(-1)
        
        # Add global context and residual
        out = out + g + residual
        return F.relu(out)

class QuoridorNet(nn.Module):
    def __init__(self, input_channels=9, hidden_channels=128, num_blocks=10):
        super().__init__()
        
        # Initial Convolution
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU()
        )
        
        # Backbone (Residual blocks with global pooling)
        self.backbone = nn.ModuleList([
            GlobalPoolingResBlock(hidden_channels) for _ in range(num_blocks)
        ])
        
        # 1. Policy Head (Output: 132 possible moves)
        # 4 pawn moves + 64 horizontal walls + 64 vertical walls
        self.policy_conv = nn.Conv2d(hidden_channels, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 17 * 17, 132)
        
        # 2. Value Head (Output: -1 to 1 win probability)
        self.value_conv = nn.Conv2d(hidden_channels, 32, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 17 * 17, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
        # 3. Distance Prediction Head (Auxiliary Target: shortest path step difference)
        self.distance_conv = nn.Conv2d(hidden_channels, 32, kernel_size=1, bias=False)
        self.distance_bn = nn.BatchNorm2d(32)
        self.distance_fc1 = nn.Linear(32 * 17 * 17, 256)
        self.distance_fc2 = nn.Linear(256, 1)
        
        # 4. Bottleneck / Ownership Head (Auxiliary Target: 17x17 heatmap)
        # Predicts the importance or ownership of each cell/intersection
        self.bottleneck_conv1 = nn.Conv2d(hidden_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bottleneck_bn = nn.BatchNorm2d(32)
        self.bottleneck_conv2 = nn.Conv2d(32, 1, kernel_size=1)
        
    def forward(self, x, action_mask=None):
        """
        x: Input tensor of shape (Batch, 7, 17, 17)
        action_mask: Optional boolean tensor of shape (Batch, 132) where False means illegal move.
                     Can be provided by the Decoupled Rule Engine.
        """
        # Backbone
        out = self.initial_conv(x)
        for block in self.backbone:
            out = block(out)
            
        # 1. Policy Head
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)
        
        # Apply action mask if provided (set illegal moves to a large negative number instead of -inf to avoid NaN in cross entropy)
        if action_mask is not None:
            # action_mask is 1 for legal, 0 for illegal
            policy_logits = policy_logits.masked_fill(~action_mask, -1e9)
            
        # 2. Value Head
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))
        
        # 3. Distance Prediction Head
        d = F.relu(self.distance_bn(self.distance_conv(out)))
        d = d.view(d.size(0), -1)
        d = F.relu(self.distance_fc1(d))
        distance = self.distance_fc2(d) # Linear output for regression (e.g. step difference)
        
        # 4. Bottleneck / Ownership Head
        b = F.relu(self.bottleneck_bn(self.bottleneck_conv1(out)))
        bottleneck_logits = self.bottleneck_conv2(b)
        bottleneck = torch.sigmoid(bottleneck_logits) # 0 to 1 heat map
        
        return policy_logits, value, distance, bottleneck

if __name__ == "__main__":
    # Quick sanity check
    model = QuoridorNet()
    # Batch size of 2, 9 channels, 17x17 grid
    dummy_input = torch.randn(2, 9, 17, 17)
    # Dummy mask: assume all moves legal except move index 0 for the first batch
    dummy_mask = torch.ones(2, 132, dtype=torch.bool)
    dummy_mask[0, 0] = False
    
    policy, value, distance, bottleneck = model(dummy_input, dummy_mask)
    
    print(f"Policy Shape: {policy.shape} (Expected: [2, 132])")
    print(f"Policy[0, 0] (illegal, should be -inf): {policy[0, 0].item()}")
    print(f"Value Shape: {value.shape} (Expected: [2, 1])")
    print(f"Distance Shape: {distance.shape} (Expected: [2, 1])")
    print(f"Bottleneck Shape: {bottleneck.shape} (Expected: [2, 1, 17, 17])")
