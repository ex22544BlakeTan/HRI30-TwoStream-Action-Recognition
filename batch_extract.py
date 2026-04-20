import os
import glob
import numpy as np
import cv2
import mediapipe as mp

# --- ⚙️ 配置区域 ---
TRAIN_FOLDER = "train_set"  
TEST_FOLDER = "test_set"    
OUTPUT_FOLDER = "skeleton_data"
OVERWRITE = True  # ⬅️ 如果设为 True，会重新提取并覆盖旧文件；设为 False 则跳过已存在的文件

# --- 初始化 MediaPipe ---
print("🔧 正在初始化 MediaPipe 模型...")
mp_pose = mp.solutions.pose
# model_complexity=1 是平衡点。如果你电脑很快，可以改成 2 (精度更高但更慢)
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

def process_one_file(video_path, save_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): 
        print(f"❌ 无法打开视频: {video_path}")
        return False
    
    frames_data = []
    
    # 🧠 [核心改进] 记录上一帧的有效数据
    # 初始化为全0，万一第一帧就没人，也没办法，只能是0
    last_valid_frame = [0] * 132 
    has_valid_frame = False

    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        frame_landmarks = []
        
        if results.pose_landmarks:
            # ✅ 检测到了人
            for lm in results.pose_landmarks.landmark:
                frame_landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
            
            # 更新“上一帧有效数据”
            last_valid_frame = frame_landmarks
            has_valid_frame = True
            frames_data.append(frame_landmarks)
        else:
            # ❌ [改进] 没检测到人 (遮挡/丢失)
            # 不要补0！使用上一帧的数据 (Forward Fill)
            if has_valid_frame:
                frames_data.append(last_valid_frame)
            else:
                # 如果视频刚开始就没人，只能先补0
                frames_data.append([0] * 132)
        
        frame_count += 1
    
    cap.release()
    
    # 保存为 .npy
    if len(frames_data) > 0:
        np.save(save_path, np.array(frames_data))
        return True
    else:
        return False

def run_batch(folder_name, split_type):
    # 找视频
    search_pattern = os.path.join(folder_name, "*.avi") # 如果有 .mp4 请改成 "*.mp4"
    files = glob.glob(search_pattern)
    print(f"\n📂 [{split_type}] 找到 {len(files)} 个视频文件")
    
    if len(files) == 0:
        print("⚠️ 警告：文件夹里是空的！")
        return

    # 创建输出目录
    save_dir = os.path.join(OUTPUT_FOLDER, split_type)
    os.makedirs(save_dir, exist_ok=True)
    
    print("🚀 开始提取...")
    count = 0
    for i, video_path in enumerate(files):
        file_id = os.path.splitext(os.path.basename(video_path))[0]
        save_path = os.path.join(save_dir, file_id + ".npy")
        
        # 检查是否需要跳过
        if os.path.exists(save_path) and not OVERWRITE:
            print(f"\r[跳过] {file_id}.npy 已存在", end="")
            continue

        # 显示进度
        print(f"\r[{i+1}/{len(files)}] 正在处理: {file_id} ... ", end="")
        
        success = process_one_file(video_path, save_path)
        if success: 
            count += 1
            
    print(f"\n🎉 {split_type} 处理完毕！成功提取 {count} 个文件。")

# --- 执行 ---
if __name__ == "__main__":
    print("-" * 30)
    print(f"当前工作目录: {os.getcwd()}")
    print("-" * 30)

    if os.path.exists(TRAIN_FOLDER):
        run_batch(TRAIN_FOLDER, "train")
    else:
        print(f"❌ 找不到训练集文件夹: {TRAIN_FOLDER}")

    if os.path.exists(TEST_FOLDER):
        run_batch(TEST_FOLDER, "test")
    else:
        print(f"❌ 找不到测试集文件夹: {TEST_FOLDER}") # 测试集可能不存在，不强制报错
        
    print("\n🏁 全部结束。请重新运行训练脚本！")