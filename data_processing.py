"""
Skeleton Extraction and Preprocessing Module.

This script utilizes MediaPipe Pose to extract 33 3D skeletal landmarks from
a batch of raw industrial interaction videos. It implements a zero-order hold 
(forward fill) strategy to robustly handle temporary occlusions or detection 
failures inherent in industrial environments, ensuring temporal continuity 
for downstream RNN/LSTM models.
"""

import os
import glob
import cv2
import numpy as np
import mediapipe as mp

# ==========================================
# Configuration
# ==========================================
TRAIN_FOLDER = "train_set"  
TEST_FOLDER = "test_set"    
OUTPUT_FOLDER = "skeleton_data"
OVERWRITE = True  # Set to False to skip processing for existing .npy files

# Initialize MediaPipe Pose model
print("[INFO] Initializing MediaPipe Pose model...")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False, 
    min_detection_confidence=0.5, 
    model_complexity=1
)

# ==========================================
# Processing Functions
# ==========================================
def process_one_file(video_path, save_path):
    """
    Processes a single video file to extract frame-by-frame skeleton data.

    Applies a forward-fill (zero-order hold) strategy for frames where the 
    subject is not detected, preserving the temporal continuity required by 
    the LSTM network.

    Args:
        video_path (str): Path to the input video file.
        save_path (str): Path to save the extracted .npy array.

    Returns:
        bool: True if extraction and saving were successful, False otherwise.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): 
        print(f"[ERROR] Cannot open video: {video_path}")
        return False
    
    frames_data = []
    
    # Forward fill initialization
    last_valid_frame = [0] * 132 
    has_valid_frame = False

    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        frame_landmarks = []
        
        if results.pose_landmarks:
            # Subject detected: extract 132 features (33 landmarks * 4 dimensions)
            for lm in results.pose_landmarks.landmark:
                frame_landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
            
            last_valid_frame = frame_landmarks
            has_valid_frame = True
            frames_data.append(frame_landmarks)
        else:
            # Detection failed (occlusion): apply forward fill
            if has_valid_frame:
                frames_data.append(last_valid_frame)
            else:
                # Fallback if the subject is missing from the very first frame
                frames_data.append([0] * 132)
        
    cap.release()
    
    if len(frames_data) > 0:
        np.save(save_path, np.array(frames_data))
        return True
    else:
        return False

def run_batch(folder_name, split_type):
    """
    Executes batch extraction over a specified directory.

    Args:
        folder_name (str): Directory containing the raw video files.
        split_type (str): Dataset split identifier ('train' or 'test').
    """
    search_pattern = os.path.join(folder_name, "*.avi")
    files = glob.glob(search_pattern)
    print(f"\n[INFO] [{split_type.upper()}] Found {len(files)} video files.")
    
    if len(files) == 0:
        print(f"[WARNING] Directory {folder_name} is empty.")
        return

    save_dir = os.path.join(OUTPUT_FOLDER, split_type)
    os.makedirs(save_dir, exist_ok=True)
    
    print("[INFO] Commencing extraction pipeline...")
    count = 0
    for i, video_path in enumerate(files):
        file_id = os.path.splitext(os.path.basename(video_path))[0]
        save_path = os.path.join(save_dir, file_id + ".npy")
        
        if os.path.exists(save_path) and not OVERWRITE:
            print(f"\r[INFO] Skipping {file_id}.npy (Already exists)", end="")
            continue

        print(f"\r[INFO] Processing [{i+1}/{len(files)}]: {file_id} ... ", end="")
        
        success = process_one_file(video_path, save_path)
        if success: 
            count += 1
            
    print(f"\n[INFO] {split_type.upper()} processing completed. Successfully extracted {count} files.")

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    print("-" * 50)
    print(f"Working Directory: {os.getcwd()}")
    print("-" * 50)

    if os.path.exists(TRAIN_FOLDER):
        run_batch(TRAIN_FOLDER, "train")
    else:
        print(f"[ERROR] Train folder not found: {TRAIN_FOLDER}")

    if os.path.exists(TEST_FOLDER):
        run_batch(TEST_FOLDER, "test")
    else:
        print(f"[WARNING] Test folder not found: {TEST_FOLDER}")
        
    print("\n[INFO] Pipeline terminated successfully. Proceed to model training.")
