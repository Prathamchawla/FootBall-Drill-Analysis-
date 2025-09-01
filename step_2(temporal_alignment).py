import json
import numpy as np
from dtw import dtw
import matplotlib.pyplot as plt

def load_json_data(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def extract_keypoint_sequences(json_data, keypoint_ids):
    """
    Extract sequences of (x, y, z) for specified keypoints across frames.
    keypoint_ids: List of MediaPipe landmark IDs (e.g., 23=left_hip, 25=left_knee, 27=left_ankle).
    Returns: np.array of shape (num_frames, len(keypoint_ids) * 3).
    """
    sequences = []
    for frame in json_data:
        keypoints = frame['player_keypoints']
        frame_data = []
        for kid in keypoint_ids:
            if str(kid) in keypoints:
                kp = keypoints[str(kid)]
                frame_data.extend([kp['x'], kp['y'], kp['z']])
            else:
                frame_data.extend([0.0, 0.0, 0.0])  # Handle missing keypoints
        sequences.append(frame_data)
    return np.array(sequences)

def align_sequences(baseline_seq, player_seq):
    """
    Perform DTW alignment between two sequences.
    Returns alignment indices (baseline_idx, player_idx) and distance.
    """
    # Define distance metric (Euclidean)
    euclidean_norm = lambda x, y: np.sqrt(np.sum((x - y) ** 2))
    
    # Perform DTW with corrected parameter
    alignment = dtw(baseline_seq, player_seq, dist_method=euclidean_norm)
    
    # Extract aligned frame indices
    aligned_indices = list(zip(alignment.index1, alignment.index2))
    return aligned_indices, alignment.distance

def main(baseline_json, player_json, output_json):
    # Load JSON data
    baseline_data = load_json_data(baseline_json)
    player_data = load_json_data(player_json)

    # Select keypoints for alignment (focus on lower body for football drills)
    keypoint_ids = [23, 24, 25, 26, 27, 28]  # left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle

    # Extract sequences
    baseline_seq = extract_keypoint_sequences(baseline_data, keypoint_ids)
    player_seq = extract_keypoint_sequences(player_data, keypoint_ids)

    # Perform DTW alignment
    aligned_indices, dtw_distance = align_sequences(baseline_seq, player_seq)

    # Save alignment results
    alignment_data = {
        'aligned_frames': [{'baseline_frame': int(b), 'player_frame': int(p)} for b, p in aligned_indices],
        'dtw_distance': float(dtw_distance)
    }
    with open(output_json, 'w') as f:
        json.dump(alignment_data, f, indent=4)

    print(f"Alignment saved to {output_json}")
    print(f"DTW Distance: {dtw_distance}")

    # Optional: Visualize alignment path
    plt.figure(figsize=(10, 5))
    plt.plot([b for b, _ in aligned_indices], [p for _, p in aligned_indices], 'b-')
    plt.xlabel('Baseline Frame')
    plt.ylabel('Player Frame')
    plt.title('DTW Alignment Path')
    plt.grid(True)
    plt.savefig('alignment_path.png')
    plt.close()
    print("Alignment path visualization saved to alignment_path.png")

# Example usage
baseline_json = 'baseline_data.json'
player_json = 'player_data.json'
output_json = 'alignment_data.json'
main(baseline_json, player_json, output_json)