"""
Two-Stream Late Fusion Inference Script.

This script executes the multimodal inference pipeline for action recognition. 
It loads pre-trained Skeleton (CNN-LSTM) and RGB (R(2+1)D) models, processes 
unseen test data (.npy and .avi), applies a late fusion strategy by weighted 
probability averaging, and generates a formatted CSV file for submission.
"""

import os
import glob
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models.video as models

# ==========================================
# Configuration
# ==========================================
SKELETON_MODEL_PATH = "best_model_v3.pth"
RGB_MODEL_PATH = "best_model_rgb.pth"

TEST_SKELETON_DIR = "skeleton_data/test"
TEST_VIDEO_DIR = "test_set"
TRAIN_CSV = "annotations/train_set_labels.csv"
OUTPUT_FILE = "test_set_labels_fusion.csv"

# Late Fusion Weights
ALPHA_RGB = 0.8
ALPHA_SKELETON = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# Model Architecture
# ==========================================
class LightweightCNNLSTM(nn.Module):
    """
    Lightweight CNN-LSTM network. Architecture must strictly match the training phase.
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
# Data Processing Pipeline
# ==========================================
def process_skeleton(npy_path):
    """
    Loads and preprocesses skeleton sequences for inference.
    Applies root centering and shoulder scaling.
    
    Args:
        npy_path (str): Path to the .npy file.
        
    Returns:
        torch.FloatTensor: Preprocessed tensor of shape [1, 100, 99].
    """
    FIXED_LENGTH = 100
    
    if not os.path.exists(npy_path): 
        return torch.zeros((1, FIXED_LENGTH, 99))
    
    raw_data = np.load(npy_path)
    if raw_data.shape[0] == 0: 
        return torch.zeros((1, FIXED_LENGTH, 99))

    frames = raw_data.shape[0]
    data = raw_data.reshape(frames, 33, 4)
    xyz = data[:, :, :3] 
    
    root = (xyz[:, 23, :] + xyz[:, 24, :]) / 2
    xyz = xyz - root.reshape(frames, 1, 3)
    
    left_shoulder = xyz[:, 11, :]
    right_shoulder = xyz[:, 12, :]
    dist = np.sqrt(np.sum((left_shoulder - right_shoulder)**2, axis=1))
    dist = np.where(dist < 1e-4, 1.0, dist).reshape(frames, 1, 1)
    xyz_norm = xyz / dist
    
    data = xyz_norm.reshape(frames, 99)

    if data.shape[0] > FIXED_LENGTH:
        start = (data.shape[0] - FIXED_LENGTH) // 2
        data = data[start : start + FIXED_LENGTH, :]
    elif data.shape[0] < FIXED_LENGTH:
        padding = np.zeros((FIXED_LENGTH - data.shape[0], 99))
        data = np.vstack((padding, data))
        
    return torch.FloatTensor(data).unsqueeze(0)

def process_video(video_path):
    """
    Loads and preprocesses RGB videos for inference.
    Applies uniform sampling and Kinetics-400 normalization.
    
    Args:
        video_path (str): Path to the .avi or .mp4 file.
        
    Returns:
        torch.FloatTensor: Preprocessed tensor of shape [1, 3, 16, 128, 128].
    """
    RESIZE_H, RESIZE_W = 128, 128
    NUM_FRAMES = 16
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, (RESIZE_W, RESIZE_H))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    finally:
        cap.release()
        
    if len(frames) == 0:
        return torch.zeros((1, 3, NUM_FRAMES, RESIZE_H, RESIZE_W))

    indices = np.linspace(0, len(frames) - 1, NUM_FRAMES).astype(int)
    sampled_frames = np.array([frames[i] for i in indices])
    
    buffer = torch.FloatTensor(sampled_frames).permute(3, 0, 1, 2) / 255.0
    
    mean = torch.tensor([0.432, 0.394, 0.376]).view(3, 1, 1, 1)
    std = torch.tensor([0.228, 0.221, 0.217]).view(3, 1, 1, 1)
    buffer = (buffer - mean) / std
    
    return buffer.unsqueeze(0)

# ==========================================
# Main Inference Execution
# ==========================================
if __name__ == "__main__":
    print(f"[INFO] Initializing Two-Stream Fusion Inference | Device: {device}")
    
    # 1. Load label mappings
    df = pd.read_csv(TRAIN_CSV, header=None)
    unique_labels = sorted(df.iloc[:, 1].unique())
    int_to_label = {i: name for i, name in enumerate(unique_labels)}
    num_classes = len(unique_labels)
    print(f"[INFO] Labels mapped successfully: {num_classes} classes identified.")

    # 2. Load Skeleton Model
    print(f"[INFO] Loading Skeleton Stream Model: {SKELETON_MODEL_PATH}")
    if not os.path.exists(SKELETON_MODEL_PATH):
        print(f"[ERROR] Skeleton model not found at {SKELETON_MODEL_PATH}")
        exit()
        
    skel_model = LightweightCNNLSTM(input_size=99, hidden_size=128, num_classes=num_classes).to(device)
    skel_model.load_state_dict(torch.load(SKELETON_MODEL_PATH, map_location=device))
    skel_model.eval()

    # 3. Load RGB Model
    print(f"[INFO] Loading RGB Stream Model: {RGB_MODEL_PATH}")
    if not os.path.exists(RGB_MODEL_PATH):
        print(f"[ERROR] RGB model not found at {RGB_MODEL_PATH}")
        exit()
        
    rgb_model = models.r2plus1d_18(weights=None)
    rgb_model.fc = nn.Linear(rgb_model.fc.in_features, num_classes)
    rgb_model.load_state_dict(torch.load(RGB_MODEL_PATH, map_location=device))
    rgb_model.to(device)
    rgb_model.eval()

    # 4. Compile test files
    test_files = glob.glob(os.path.join(TEST_VIDEO_DIR, "*.avi"))
    if len(test_files) == 0: 
        test_files = glob.glob(os.path.join(TEST_VIDEO_DIR, "*.mp4"))
    
    print(f"[INFO] Discovered {len(test_files)} test samples.")
    print(f"[INFO] Applied Fusion Strategy: RGB({ALPHA_RGB}) + Skeleton({ALPHA_SKELETON})")
    
    results = []
    
    with torch.no_grad():
        for i, video_path in enumerate(test_files):
            file_id = os.path.splitext(os.path.basename(video_path))[0]
            video_name = file_id + ".avi"
            
            npy_path = os.path.join(TEST_SKELETON_DIR, file_id + ".npy")
            
            # Stream 1: Skeleton Forward Pass
            skel_input = process_skeleton(npy_path).to(device)
            skel_logits = skel_model(skel_input)
            skel_probs = torch.softmax(skel_logits, dim=1) 
            
            # Stream 2: RGB Forward Pass
            rgb_input = process_video(video_path).to(device)
            rgb_logits = rgb_model(rgb_input)
            rgb_probs = torch.softmax(rgb_logits, dim=1)   
            
            # Late Fusion computation
            final_probs = (ALPHA_RGB * rgb_probs) + (ALPHA_SKELETON * skel_probs)
            
            _, predicted = torch.max(final_probs, 1)
            label_name = int_to_label[predicted.item()]
            
            results.append([video_name, label_name])
            
            if (i+1) % 50 == 0: 
                print(f"  [INFO] Processed {i+1}/{len(test_files)} samples...")

    # 5. Export results
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_FILE, index=False, header=False)
    
    print(f"\n[INFO] Multimodal inference completed successfully.")
    print(f"[INFO] Submission file generated: {os.path.abspath(OUTPUT_FILE)}")
