"""
Skeleton Video Stream Training Pipeline.

This script trains a Lightweight CNN-LSTM-Attention network on 3D skeleton sequences 
for action recognition. It includes robust data preprocessing (root centering, shoulder 
scaling), dynamic sequence padding, and on-the-fly data augmentations (Gaussian noise 
and random limb masking) to mitigate overfitting. It also utilizes class weight 
balancing to handle imbalanced datasets typical in industrial environments.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight

# ==========================================
# Configuration
# ==========================================
CSV_FILE = "annotations/train_set_labels.csv"
DATA_FOLDER = "skeleton_data/train"

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 80
FIXED_LENGTH = 100
INPUT_SIZE = 99     # Flattened 3D coordinates (33 landmarks * 3 axes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# Dataset Processing
# ==========================================
class SkeletonDataset(Dataset):
    """
    Custom Dataset for loading and augmenting 3D skeleton sequences.

    Args:
        csv_path (str): Path to the annotation CSV file.
        data_dir (str): Directory containing extracted .npy skeleton features.
        sequence_length (int): Fixed temporal length for padding/truncating.
        augment (bool): Whether to apply random data augmentations during training.
    """
    def __init__(self, csv_path, data_dir, sequence_length, augment=False):
        self.labels_df = pd.read_csv(csv_path, header=None)
        self.data_dir = data_dir
        self.seq_len = sequence_length
        self.augment = augment
        
        unique_labels = sorted(self.labels_df.iloc[:, 1].unique())
        self.label_to_int = {name: i for i, name in enumerate(unique_labels)}
        self.all_labels = [self.label_to_int[label] for label in self.labels_df.iloc[:, 1]]

    def __len__(self):
        return len(self.labels_df)

    def normalize_skeleton(self, raw_data):
        """
        Applies spatial normalization to the skeleton sequence.
        Includes root (hip) centering and shoulder-width distance scaling.
        """
        frames = raw_data.shape[0]
        data = raw_data.reshape(frames, 33, 4)
        xyz = data[:, :, :3]
        
        # Root Centering (Mid-hip alignment)
        root = (xyz[:, 23, :] + xyz[:, 24, :]) / 2
        xyz = xyz - root.reshape(frames, 1, 3)
        
        # Shoulder Scaling
        left_shoulder = xyz[:, 11, :]
        right_shoulder = xyz[:, 12, :]
        dist = np.sqrt(np.sum((left_shoulder - right_shoulder)**2, axis=1))
        dist = np.where(dist < 1e-4, 1.0, dist).reshape(frames, 1, 1)
        xyz_norm = xyz / dist
        
        return xyz_norm.reshape(frames, INPUT_SIZE)

    def apply_augmentation(self, data):
        """
        Applies stochastic data augmentations: Gaussian noise and random limb masking.
        """
        # Inject Gaussian Noise
        if random.random() > 0.5: 
            data = data + np.random.normal(0, 0.002, data.shape)
            
        # Random Limb Masking (Simulating occlusions)
        if random.random() > 0.6: 
            temp = data.reshape(-1, 33, 3)
            if random.random() > 0.5:
                # Mask left arm
                temp[:, [11, 13, 15], :] = 0
            else:
                # Mask right arm
                temp[:, [12, 14, 16], :] = 0
            data = temp.reshape(-1, INPUT_SIZE)
            
        return data

    def __getitem__(self, idx):
        file_name_avi = self.labels_df.iloc[idx, 0]
        file_id = os.path.splitext(file_name_avi)[0]
        npy_path = os.path.join(self.data_dir, file_id + ".npy")
        label = self.label_to_int[self.labels_df.iloc[idx, 1]]
        
        if os.path.exists(npy_path):
            raw_data = np.load(npy_path)
            data = self.normalize_skeleton(raw_data) if raw_data.shape[0] > 0 else np.zeros((self.seq_len, INPUT_SIZE))
        else:
            data = np.zeros((self.seq_len, INPUT_SIZE))

        # Temporal Padding / Truncating
        current_len = data.shape[0]
        if current_len > self.seq_len:
            start = (current_len - self.seq_len) // 2
            data = data[start : start + self.seq_len, :]
        elif current_len < self.seq_len:
            padding = np.zeros((self.seq_len - current_len, INPUT_SIZE))
            data = np.vstack((padding, data))

        if self.augment:
            data = self.apply_augmentation(data)

        return torch.FloatTensor(data), torch.tensor(label, dtype=torch.long)

# ==========================================
# Model Architecture
# ==========================================
class LightweightCNNLSTM(nn.Module):
    """
    Lightweight spatio-temporal architecture for skeleton classification.
    
    Architecture:
        - 1D-CNN: Extracts local temporal-spatial features without reducing sequence length.
        - Bi-LSTM: Captures bidirectional temporal dependencies.
        - Multihead Attention: Aggregates global contextual information.
    """
    def __init__(self, input_size, hidden_size, num_classes):
        super(LightweightCNNLSTM, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1) 
        )
        
        self.lstm = nn.LSTM(64, hidden_size, num_layers=2, 
                            batch_first=True, bidirectional=True, dropout=0.3)
        
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size*2, num_heads=4, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        c_in = x.permute(0, 2, 1) 
        c_out = self.cnn(c_in)    
        lstm_in = c_out.permute(0, 2, 1) 
        
        lstm_out, _ = self.lstm(lstm_in)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        pooled = torch.mean(attn_out, dim=1)
        
        out = self.fc(pooled)
        return out

# ==========================================
# Main Training Routine
# ==========================================
if __name__ == "__main__":
    print(f"[INFO] Initializing Skeleton Stream Training Pipeline | Device: {device}")
    
    # 1. Dataset Preparation
    train_full = SkeletonDataset(CSV_FILE, DATA_FOLDER, FIXED_LENGTH, augment=True)
    val_full = SkeletonDataset(CSV_FILE, DATA_FOLDER, FIXED_LENGTH, augment=False)
    
    # Compute Class Weights for imbalanced dataset handling
    y_train = train_full.all_labels
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"[INFO] Class weights computed and loaded successfully.")

    # Data splitting
    dataset_len = len(train_full)
    indices = list(range(dataset_len))
    split = int(0.2 * dataset_len)
    np.random.seed(42)
    np.random.shuffle(indices)
    train_idx, val_idx = indices[split:], indices[:split]
    
    train_loader = DataLoader(train_full, batch_size=BATCH_SIZE, 
                              sampler=torch.utils.data.SubsetRandomSampler(train_idx))
    val_loader = DataLoader(val_full, batch_size=BATCH_SIZE, 
                            sampler=torch.utils.data.SubsetRandomSampler(val_idx))
    
    # 2. Model Initialization
    num_classes = len(train_full.label_to_int)
    model = LightweightCNNLSTM(input_size=INPUT_SIZE, hidden_size=128, num_classes=num_classes).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model initialized. Total trainable parameters: {total_params:,}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    # 3. Training Loop
    print("[INFO] Commencing Training Loop...")
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        # Validation Phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        
        scheduler.step(val_acc)
        lr = optimizer.param_groups[0]['lr']
        
        print(f"[INFO] Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {lr:.6f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model_v3.pth")
            print(f"  [SAVE] New best model checkpoint saved: {best_acc:.2f}%")

    print(f"\n[INFO] Training Pipeline Terminated. Best Validation Accuracy: {best_acc:.2f}%")
