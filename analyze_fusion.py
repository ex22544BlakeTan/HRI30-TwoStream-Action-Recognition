import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import torchvision.models.video as models
from torch.utils.data import Dataset, DataLoader

# =========================================================================
# ⚙️ 配置区域
# =========================================================================
SKELETON_MODEL_PATH = "best_model_v3.pth"
RGB_MODEL_PATH = "best_model_rgb.pth"

# 注意：这里我们使用【训练集】的路径来进行验证
TRAIN_VIDEO_DIR = "train_set"
TRAIN_SKELETON_DIR = "skeleton_data/train"
CSV_FILE = "annotations/train_set_labels.csv"

# 融合权重
ALPHA_RGB = 0.8
ALPHA_SKELETON = 0.2

BATCH_SIZE = 8  # 验证时不反向传播，可以稍微大点
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================================
# 🏗️ 模型定义 & 数据处理 (复用之前的逻辑)
# =========================================================================

# --- 1. 骨架模型 ---
class LightweightCNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LightweightCNNLSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.1))
        self.lstm = nn.LSTM(64, hidden_size, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size*2, num_heads=4, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        c_in = x.permute(0, 2, 1)
        c_out = self.cnn(c_in)
        lstm_in = c_out.permute(0, 2, 1)
        lstm_out, _ = self.lstm(lstm_in)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        pooled = torch.mean(attn_out, dim=1)
        return self.fc(pooled)

# --- 2. 双模态数据集类 ---
class DualStreamDataset(Dataset):
    def __init__(self, csv_path, video_dir, skeleton_dir):
        self.labels_df = pd.read_csv(csv_path, header=None)
        self.video_dir = video_dir
        self.skeleton_dir = skeleton_dir
        self.unique_labels = sorted(self.labels_df.iloc[:, 1].unique())
        self.label_to_int = {name: i for i, name in enumerate(self.unique_labels)}

    def __len__(self):
        return len(self.labels_df)

    def load_skeleton(self, file_id):
        # 骨架处理逻辑 (train_model_v3.py)
        path = os.path.join(self.skeleton_dir, file_id + ".npy")
        if not os.path.exists(path): return torch.zeros((100, 99))
        raw = np.load(path)
        if raw.shape[0] == 0: return torch.zeros((100, 99))
        
        # Norm
        frames = raw.shape[0]
        data = raw.reshape(frames, 33, 4)[:, :, :3]
        root = (data[:, 23, :] + data[:, 24, :]) / 2
        data = data - root.reshape(frames, 1, 3)
        ls, rs = data[:, 11, :], data[:, 12, :]
        dist = np.sqrt(np.sum((ls - rs)**2, axis=1)).reshape(frames, 1, 1)
        dist = np.where(dist < 1e-4, 1.0, dist)
        data = (data / dist).reshape(frames, 99)
        
        # Pad
        if data.shape[0] > 100: data = data[(data.shape[0]-100)//2 : (data.shape[0]-100)//2+100]
        elif data.shape[0] < 100: data = np.vstack((np.zeros((100-data.shape[0], 99)), data))
        
        return torch.FloatTensor(data)

    def load_video(self, file_id):
        # RGB处理逻辑 (rgb_model.py)
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
            
        if len(frames) == 0: return torch.zeros((3, 16, 128, 128))
        
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

# =========================================================================
# 🚀 主程序：验证融合效果
# =========================================================================
if __name__ == "__main__":
    print(f"📊 启动本地验证 (权重搜索版) | 设备: {device}")
    
    # 1. 准备验证集 (随机采样 20%)
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

    # 2. 加载模型
    print("🧠 加载模型...")
    skel_model = LightweightCNNLSTM(99, 128, num_classes).to(device)
    skel_model.load_state_dict(torch.load(SKELETON_MODEL_PATH, map_location=device))
    skel_model.eval()
    
    rgb_model = models.r2plus1d_18(weights=None)
    rgb_model.fc = nn.Linear(rgb_model.fc.in_features, num_classes)
    rgb_model.load_state_dict(torch.load(RGB_MODEL_PATH, map_location=device))
    rgb_model.to(device)
    rgb_model.eval()

    # 3. 收集所有验证样本的概率 (不进行 argmax)
    print("🔥 正在收集预测概率...")
    all_skel_probs = []
    all_rgb_probs = []
    all_labels = []
    
    with torch.no_grad():
        for i, (skel_in, rgb_in, labels) in enumerate(val_loader):
            skel_in, rgb_in, labels = skel_in.to(device), rgb_in.to(device), labels.to(device)
            
            # 获取概率分布
            skel_probs = torch.softmax(skel_model(skel_in), dim=1)
            rgb_probs = torch.softmax(rgb_model(rgb_in), dim=1)
            
            all_skel_probs.append(skel_probs.cpu())
            all_rgb_probs.append(rgb_probs.cpu())
            all_labels.append(labels.cpu())
            
            if (i+1) % 10 == 0: print(f"  Batch {i+1} done...")

    # 拼接
    final_skel_probs = torch.cat(all_skel_probs)
    final_rgb_probs = torch.cat(all_rgb_probs)
    final_labels = torch.cat(all_labels)
    
    print("-" * 60)
    print("⚖️ 开始寻找最佳融合权重...")
    
    best_acc = 0.0
    best_alpha = 0.0
    best_preds = None
    
    # 4. 网格搜索 (从 0.0 到 1.0，步长 0.01)
    # alpha 是 RGB 的权重，(1-alpha) 是骨架的权重
    for alpha in np.linspace(0, 1, 101):
        # 融合公式
        fusion_probs = (alpha * final_rgb_probs) + ((1 - alpha) * final_skel_probs)
        
        _, preds = torch.max(fusion_probs, 1)
        acc = (preds == final_labels).float().mean().item() * 100
        
        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha
            best_preds = preds
            
    print("-" * 60)
    print(f"🏆 最佳融合结果:")
    print(f"   最佳 RGB 权重 (Alpha): {best_alpha:.2f}")
    print(f"   最佳 Skeleton 权重:    {1 - best_alpha:.2f}")
    print(f"   🚀 最高准确率:         {best_acc:.2f}%")
    print("-" * 60)
    
    # 5. 用最佳结果画图
    cm = confusion_matrix(final_labels.numpy(), best_preds.numpy())
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap='Blues', 
                xticklabels=full_dataset.unique_labels, 
                yticklabels=full_dataset.unique_labels)
    plt.title(f'Optimized Fusion Matrix (Alpha={best_alpha:.2f}, Acc={best_acc:.2f}%)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('fusion_analysis_optimized.png')
    print("✅ 优化后的混淆矩阵已保存为 fusion_analysis_optimized.png")