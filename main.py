import cv2
import mediapipe as mp
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# MediaPipeのセットアップ
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 動画の読み込み
reference_video_path = 'reference_video.mp4'  # 動画①のパスを指定
comparison_video_path = 'comparison_video.mp4'  # 動画②のパスを指定
output_video_path = 'output_comparison_video.mp4'

cap_ref = cv2.VideoCapture(reference_video_path)
cap_cmp = cv2.VideoCapture(comparison_video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None

fps = cap_ref.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps / 2)  # 毎秒2回検出

def get_pose_landmarks(frame):
    # RGBに変換
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # MediaPipeでポーズを検出
    results = pose.process(image)
    if results.pose_landmarks:
        return results.pose_landmarks
    return None

def extract_landmark_coords(landmarks):
    return [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]

def calculate_dtw_similarity(coords1, coords2):
    if not coords1 or not coords2:
        return 0
    distance, _ = fastdtw(coords1, coords2, dist=euclidean)
    similarity = np.exp(-distance)
    return similarity * 100  # パーセンテージに変換

frame_count = 0

while cap_ref.isOpened() and cap_cmp.isOpened():
    ret_ref, frame_ref = cap_ref.read()
    ret_cmp, frame_cmp = cap_cmp.read()
    
    if not ret_ref or not ret_cmp:
        break
    
    # フレームの高さと幅を取得
    h, w = frame_ref.shape[:2]
    
    # 初回のフレームでVideoWriterを初期化
    if out is None:
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
    
    if frame_count % frame_interval == 0:
        # ポーズのランドマークを取得
        landmarks_ref = get_pose_landmarks(frame_ref)
        landmarks_cmp = get_pose_landmarks(frame_cmp)
        
        # ランドマークの座標を抽出
        coords_ref = extract_landmark_coords(landmarks_ref) if landmarks_ref else []
        coords_cmp = extract_landmark_coords(landmarks_cmp) if landmarks_cmp else []
        
        # DTWを使って類似度を計算
        similarity = calculate_dtw_similarity(coords_ref, coords_cmp)
        
        # 類似度をフレームに表示
        cv2.putText(frame_cmp, f'Similarity: {similarity:.2f}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # ランドマークを描画
        if landmarks_cmp:
            mp_drawing.draw_landmarks(frame_cmp, landmarks_cmp, mp_pose.POSE_CONNECTIONS)
    
    # フレームを書き込み
    out.write(frame_cmp)
    
    # 結果のフレームを表示（オプション）
    cv2.imshow('Dance Comparison', frame_cmp)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    frame_count += 1

# リソースを解放
cap_ref.release()
cap_cmp.release()
out.release()
cv2.destroyAllWindows()
