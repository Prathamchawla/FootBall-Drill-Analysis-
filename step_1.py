import cv2
import mediapipe as mp
import numpy as np
import json
from ultralytics import YOLO
from sort import Sort

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)

# Initialize YOLO model (replace with your trained model path)
yolo_model = YOLO('models/best.pt')  # e.g., 'runs/detect/ball_cone_detector/weights/best.pt'

# Initialize SORT tracker
tracker = Sort()

def process_video(input_video_path, output_video_path, output_json_path):
    # Open video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Store all frame data for JSON
    frame_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1. Pose Estimation with MediaPipe
        pose_results = pose.process(image_rgb)
        keypoints = {}
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
                keypoints[idx] = {
                    'x': float(landmark.x),
                    'y': float(landmark.y),
                    'z': float(landmark.z),
                    'visibility': float(landmark.visibility)
                }

        # 2. Object Detection with YOLO
        yolo_results = yolo_model(frame,conf=0.6,iou=0.3)[0]
        detections = []
        for det in yolo_results.boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            conf = float(det.conf[0])
            cls = int(det.cls[0])
            label = yolo_model.names[cls]
            detections.append([x1, y1, x2, y2, conf])
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 3. Object Tracking with SORT
        detections = np.array(detections) if detections else []
        trackers = tracker.update(detections)
        tracked_objects = []
        for track in trackers:
            x1, y1, x2, y2, track_id = map(int, track)
            label = 'ball' if any(det[4] > 0.5 and yolo_model.names[int(yolo_results.boxes.cls[i])] == 'ball' 
                                 for i, det in enumerate(detections) if np.allclose(det[:4], track[:4], atol=5)) else 'cone'
            tracked_objects.append({
                'track_id': track_id,
                'class': label,
                'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
            })
            # Draw track ID
            cv2.putText(frame, f'ID: {track_id}', (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Store frame data
        frame_data.append({
            'frame': len(frame_data),
            'player_keypoints': keypoints,
            'objects': tracked_objects
        })

        # Write frame to output video
        out.write(frame)

        # Display frame (optional, comment out if not needed)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Save to JSON
    with open(output_json_path, 'w') as f:
        json.dump(frame_data, f, indent=4)
    print(f"Processed video saved to {output_video_path}")
    print(f"Data saved to {output_json_path}")

# Example usage
input_video = r'/home/dire/Desktop/Drill Analysis/practise-sample1.mp4'  # Coach video
output_video = 'player_with_detections.mp4'
output_json = 'player_data.json'
process_video(input_video, output_video, output_json)