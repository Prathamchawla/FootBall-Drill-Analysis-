import json
import numpy as np
import matplotlib.pyplot as plt
from math import atan2, degrees

def load_json_data(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def calculate_angle(p1, p2, p3):
    """
    Calculate angle (in degrees) between three points (p1-p2-p3).
    p1, p2, p3: Dict with 'x', 'y' coordinates (normalized).
    """
    v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y']])
    v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y']])
    dot_product = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norms == 0:
        return 0.0
    angle = np.arccos(dot_product / norms)
    return degrees(angle)

def compute_joint_angles(keypoints, joint_triplets):
    """
    Compute angles for specified joint triplets (e.g., hip-knee-ankle).
    joint_triplets: List of [id1, id2, id3] (e.g., [23, 25, 27] for left_hip-knee-ankle).
    Returns: Dict of angles {triplet_idx: angle}.
    """
    angles = {}
    for idx, (id1, id2, id3) in enumerate(joint_triplets):
        if str(id1) in keypoints and str(id2) in keypoints and str(id3) in keypoints:
            p1, p2, p3 = keypoints[str(id1)], keypoints[str(id2)], keypoints[str(id3)]
            if p1['visibility'] > 0.5 and p2['visibility'] > 0.5 and p3['visibility'] > 0.5:
                angles[idx] = calculate_angle(p1, p2, p3)
            else:
                angles[idx] = 0.0  # Handle low-visibility keypoints
        else:
            angles[idx] = 0.0
    return angles

def check_drill_completion(baseline_objects, player_objects, aligned_frames):
    """
    Check if player interacted with all cones and ball as in baseline.
    Returns: List of completed actions and missing actions.
    """
    baseline_cone_ids = set()
    player_cone_ids = set()
    ball_interaction = {'baseline': False, 'player': False}

    for frame in baseline_objects:
        for obj in frame['objects']:
            if obj['class'] == 'cone':
                baseline_cone_ids.add(obj['track_id'])
            if obj['class'] == 'ball':
                ball_interaction['baseline'] = True

    for frame in player_objects:
        for obj in frame['objects']:
            if obj['class'] == 'cone':
                player_cone_ids.add(obj['track_id'])
            if obj['class'] == 'ball':
                ball_interaction['player'] = True

    completed_cones = baseline_cone_ids.intersection(player_cone_ids)
    missing_cones = baseline_cone_ids - player_cone_ids
    ball_completed = ball_interaction['baseline'] == ball_interaction['player']

    return {
        'completed_cones': list(completed_cones),
        'missing_cones': list(missing_cones),
        'ball_interaction': ball_completed
    }

def main(baseline_json, player_json, alignment_json, output_json):
    # Load data
    baseline_data = load_json_data(baseline_json)
    player_data = load_json_data(player_json)
    alignment_data = load_json_data(alignment_json)

    # Define joint triplets for angle calculation (hip-knee-ankle for both legs)
    joint_triplets = [
        [23, 25, 27],  # left_hip-knee-ankle
        [24, 26, 28]   # right_hip-knee-ankle
    ]

    # Compute angles for aligned frames
    angle_diffs = []
    for pair in alignment_data['aligned_frames']:
        b_frame = pair['baseline_frame']
        p_frame = pair['player_frame']
        
        if b_frame < len(baseline_data) and p_frame < len(player_data):
            b_keypoints = baseline_data[b_frame]['player_keypoints']
            p_keypoints = player_data[p_frame]['player_keypoints']
            
            b_angles = compute_joint_angles(b_keypoints, joint_triplets)
            p_angles = compute_joint_angles(p_keypoints, joint_triplets)
            
            # Calculate angle differences
            frame_diff = []
            for idx in b_angles:
                diff = abs(b_angles[idx] - p_angles[idx])
                frame_diff.append(diff)
            angle_diffs.append(frame_diff)

    # Compute form accuracy (average angle difference)
    angle_diffs = np.array(angle_diffs)
    form_accuracy = {
        'left_leg': float(np.mean(angle_diffs[:, 0])) if len(angle_diffs) > 0 else 0.0,
        'right_leg': float(np.mean(angle_diffs[:, 1])) if len(angle_diffs) > 0 else 0.0
    }
    overall_form_accuracy = float(np.mean(angle_diffs)) if len(angle_diffs) > 0 else 0.0

    # Timing consistency (use DTW distance and frame offsets)
    frame_offsets = [abs(b - p) for b, p in [(pair['baseline_frame'], pair['player_frame']) 
                                            for pair in alignment_data['aligned_frames']]]
    timing_consistency = {
        'avg_frame_offset': float(np.mean(frame_offsets)) if frame_offsets else 0.0,
        'dtw_distance': float(alignment_data['dtw_distance'])
    }

    # Drill completion
    drill_completion = check_drill_completion(baseline_data, player_data, alignment_data['aligned_frames'])

    # Save results
    results = {
        'form_accuracy': {
            'left_leg_angle_diff': form_accuracy['left_leg'],
            'right_leg_angle_diff': form_accuracy['right_leg'],
            'overall_angle_diff': overall_form_accuracy
        },
        'timing_consistency': timing_consistency,
        'drill_completion': drill_completion
    }
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Analysis results saved to {output_json}")

    # Visualize angle differences
    plt.figure(figsize=(10, 5))
    plt.plot(angle_diffs[:, 0], label='Left Leg Angle Diff', color='blue')
    plt.plot(angle_diffs[:, 1], label='Right Leg Angle Diff', color='red')
    plt.xlabel('Aligned Frame Pair')
    plt.ylabel('Angle Difference (degrees)')
    plt.title('Joint Angle Differences')
    plt.legend()
    plt.grid(True)
    plt.savefig('angle_differences.png')
    plt.close()
    print("Angle differences plot saved to angle_differences.png")

# Example usage
baseline_json = 'baseline_data.json'
player_json = 'player_data.json'
alignment_json = 'alignment_data.json'
output_json = 'movement_analysis.json'
main(baseline_json, player_json, alignment_json,output_json)