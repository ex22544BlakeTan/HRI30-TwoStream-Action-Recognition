"""
Two-Stream Multi-modal Fusion Evaluation Script.

This script evaluates a Late Fusion strategy using pre-trained RGB (3D-CNN) 
and Skeleton (CNN-LSTM) models. It performs a grid search to determine the optimal 
fusion weights and generates a confusion matrix for the local validation set.
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.models.video as models
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader

# ==========================================
# Configuration
# ==========================================
SKELETON_MODEL_PATH = "best_model_v3.pth"
RGB_MODEL_PATH = "best_model_rgb.pth"
TRAIN_VIDEO_DIR = "train_set"
TRAIN_SKELETON_DIR = "skeleton_data/train"
CSV_FILE = "annotations/train_set_labels.csv"

BATCH_SIZE = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# Model Definition
# ==========================================
class LightweightCNNLSTM(nn.Module):
    """
    Lightweight CNN-LSTM network for skeleton-based action recognition.

    Args:
        input_size (int): Dimension of the input skeleton features.
        hidden_size (int): Number of features in the LSTM hidden state.
        num_classes (int): Number of output action classes.
    """
    def __init__(self, input_size, hidden_size, num_classes):
        super(LightweightCNNLSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), 
            nn.ReLU(), 
            nn.Dropout(0.1)
        )
        self.lstm = nn.LSTM(64, hidden_size, num_layers=2, batch_first=True, 
                            bidirectional=True, dropout=0.3)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size*2, 
                                               num_heads=4, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        c_in = x.permute(0, 2, 1)
        c_out = self.cnn(c_in)
        lstm_in = c_out.permute(0, 2, 1)
        lstm_out, _ = self.lstm(lstm_in)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        pooled = torch.mean(attn_out, dim=1)
        return self.fc(pooled)

# ==========================================
# Dataset Processing
# ==========================================
class DualStreamDataset(Dataset):
    """
    Custom Dataset to load paired skeleton sequences and RGB video frames.

    Args:
        csv_path (str): Path to the annotation CSV file.
        video_dir (str): Directory containing raw .avi videos.
        skeleton_dir (str): Directory containing extracted .npy skeleton features.
    """
    def __init__(self, csv_path, video_dir, skeleton_dir):
        self.labels_df = pd.read_csv(csv_path, header=None)
        self.video_dir = video_dir
        self.skeleton_dir = skeleton_dir
        self.unique_labels = sorted(self.labels_df.iloc[:, 1].unique())
        self.label_to_int = {name: i for i, name in enumerate(self.unique_labels)}

    def __len__(self):
        return len(self.labels_df)

    def load_skeleton(self, file_id):
        """
        Loads and normalizes skeleton data, applying root centering and padding.
        """
        path = os.path.join(self.skeleton_dir, file_id + ".npy")
        if not os.path.exists(path): 
            return torch.zeros((100, 99))
        
        raw = np.load(path)
        if raw.shape[0] == 0: 
            return torch.zeros((100, 99))
        
        frames = raw.shape[0]
        data = raw.reshape(frames, 33, 4)[:, :, :3]
        
        # Root centering
        root = (data[:, 23, :] + data[:, 24, :]) / 2
        data = data - root.reshape(frames, 1, 3)
        
        # Shoulder scaling
        ls, rs = data[:, 11, :], data[:, 12, :]
        dist = np.sqrt(np.sum((ls - rs)**2, axis=1)).reshape(frames, 1, 1)
        dist = np.where(dist < 1e-4, 1.0, dist)
        data = (data / dist).reshape(frames, 99)
        
        # Padding/Truncation to fixed length (100)
        if data.shape[0] > 100: 
            data = data[(data.shape[0]-100)//2 : (data.shape[0]-100)//2+100]
        elif data.shape[0] < 100: 
            data = np.vstack((np.zeros((100-data.shape[0], 99)), data))
        
        return torch.FloatTensor(data)

    def load_video(self, file_id):
        """
        Loads video, uniformly samples 16 frames, and applies Kinetics-400 normalization.
        """
        path = os.path.join(self.video_dir, file_id + ".avi")
        cap = cv2.VideoCapture(path)
        frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.resize(frame, (128, 128))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        finally:
            cap.release()
            
        if len(frames) == 0: 
            return torch.zeros((3, 16, 128, 128))
        
        indices = np.linspace(0, len(frames)-1, 16).astype(int)
        buffer = torch.FloatTensor(np.array([frames[i] for i in indices])).permute(3, 0, 1, 2) / 255.0
        
        mean = torch.tensor([0.432, 0.394, 0.376]).view(3, 1, 1, 1)
        std = torch.tensor([0.228, 0.221, 0.217]).view(3, 1, 1, 1)
        return (buffer - mean) / std

    def __getitem__(self, idx):
        fname = self.labels_df.iloc[idx, 0]
        fid = os.path.splitext(fname)[0]
        label = self.label_to_int[self.labels_df.iloc[idx, 1]]
        
        skel = self.load_skeleton(fid)
        rgb = self.load_video(fid)
        
        return skel, rgb, label

# ==========================================
# Main Evaluation Routine
# ==========================================
if __name__ == "__main__":
    print(f"Starting local validation | Device: {device}")
    
    full_dataset = DualStreamDataset(CSV_FILE, TRAIN_VIDEO_DIR, TRAIN_SKELETON_DIR)
    dataset_len = len(full_dataset)
    indices = list(range(dataset_len))
    
    np.random.seed(42) 
    np.random.shuffle(indices)
    split = int(0.8 * dataset_len)
    val_indices = indices[split:]
    
    val_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, 
                            sampler=torch.utils.data.SubsetRandomSampler(val_indices),
                            num_workers=0)
    
    num_classes = len(full_dataset.unique_labels)

    print("Loading models...")
    skel_model = LightweightCNNLSTM(99, 128, num_classes).to(device)
    skel_model.load_state_dict(torch.load(SKELETON_MODEL_PATH, map_location=device))
    skel_model.eval()
    
    rgb_model = models.r2plus1d_18(weights=None)
    rgb_model.fc = nn.Linear(rgb_model.fc.in_features, num_classes)
    rgb_model.load_state_dict(torch.load(RGB_MODEL_PATH, map_location=device))
    rgb_model.to(device)
    rgb_model.eval()

    print("Collecting prediction probabilities...")
    all_skel_probs = []
    all_rgb_probs = []
    all_labels = []
    
    with torch.no_grad():
        for i, (skel_in, rgb_in, labels) in enumerate(val_loader):
            skel_in, rgb_in, labels = skel_in.to(device), rgb_in.to(device), labels.to(device)
            
            skel_probs = torch.softmax(skel_model(skel_in), dim=1)
            rgb_probs = torch.softmax(rgb_model(rgb_in), dim=1)
            
            all_skel_probs.append(skel_probs.cpu())
            all_rgb_probs.append(rgb_probs.cpu())
            all_labels.append(labels.cpu())
            
            if (i+1) % 10 == 0: 
                print(f"  Processed batch {i+1}")

    final_skel_probs = torch.cat(all_skel_probs)
    final_rgb_probs = torch.cat(all_rgb_probs)
    final_labels = torch.cat(all_labels)
    
    print("-" * 60)
    print("Initiating grid search for optimal fusion weights...")
    
    best_acc = 0.0
    best_alpha = 0.0
    best_preds = None
    
    for alpha in np.linspace(0, 1, 101):
        fusion_probs = (alpha * final_rgb_probs) + ((1 - alpha) * final_skel_probs)
        _, preds = torch.max(fusion_probs, 1)
        acc = (preds == final_labels).float().mean().item() * 100
        
        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha
            best_preds = preds
            
    print("-" * 60)
    print("Optimal Fusion Results:")
    print(f"   RGB Weight (Alpha): {best_alpha:.2f}")
    print(f"   Skeleton Weight:    {1 - best_alpha:.2f}")
    print(f"   Max Accuracy:       {best_acc:.2f}%")
    print("-" * 60)
    
    cm = confusion_matrix(final_labels.numpy(), best_preds.numpy())
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap='Blues', 
                xticklabels=full_dataset.unique_labels, 
                yticklabels=full_dataset.unique_labels)
    plt.title(f'Optimized Fusion Matrix (Alpha={best_alpha:.2f}, Acc={best_acc:.2f}%)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('fusion_analysis_optimized.png')
    print("Confusion matrix saved to fusion_analysis_optimized.png")
