import json
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def load_json_data(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def create_ghost_overlay(baseline_video, player_video, alignment_data, output_video):
    # Open videos
    baseline_cap = cv2.VideoCapture(baseline_video)
    player_cap = cv2.VideoCapture(player_video)
    if not (baseline_cap.isOpened() and player_cap.isOpened()):
        print("Error opening video files")
        return

    # Get video properties
    frame_width = int(player_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(player_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = player_cap.get(cv2.CAP_PROP_FPS)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    # Define MediaPipe pose connections
    pose_connections = mp_pose.POSE_CONNECTIONS

    # Process aligned frames
    for pair in alignment_data['aligned_frames']:
        b_frame = pair['baseline_frame']
        p_frame = pair['player_frame']

        # Read frames
        baseline_cap.set(cv2.CAP_PROP_POS_FRAMES, b_frame)
        player_cap.set(cv2.CAP_PROP_POS_FRAMES, p_frame)
        ret_b, b_frame_img = baseline_cap.read()
        ret_p, p_frame_img = player_cap.read()
        if not (ret_b and ret_p):
            continue

        # Create ghost overlay (player frame as base, coach as semi-transparent)
        overlay = p_frame_img.copy()
        alpha = 0.4  # Transparency for coach's skeleton
        coach_skeleton = np.zeros_like(overlay)

        # Draw coach's skeleton (blue)
        baseline_data = load_json_data('baseline_data.json')
        if str(b_frame) in [str(f['frame']) for f in baseline_data]:
            b_keypoints = next(f['player_keypoints'] for f in baseline_data if f['frame'] == b_frame)
            # Draw landmarks
            for idx, kp in b_keypoints.items():
                if kp['visibility'] < 0.5:  # Skip low-visibility keypoints
                    continue
                x, y = int(kp['x'] * frame_width), int(kp['y'] * frame_height)
                cv2.circle(coach_skeleton, (x, y), 5, (255, 0, 0), -1)  # Blue dots
            # Draw connections
            for connection in pose_connections:
                start_idx, end_idx = connection
                if str(start_idx) in b_keypoints and str(end_idx) in b_keypoints:
                    start_kp = b_keypoints[str(start_idx)]
                    end_kp = b_keypoints[str(end_idx)]
                    if start_kp['visibility'] > 0.5 and end_kp['visibility'] > 0.5:
                        start_point = (int(start_kp['x'] * frame_width), int(start_kp['y'] * frame_height))
                        end_point = (int(end_kp['x'] * frame_width), int(end_kp['y'] * frame_height))
                        cv2.line(coach_skeleton, start_point, end_point, (255, 0, 0), 2)

        # Overlay coach's skeleton on player's frame
        overlay = cv2.addWeighted(coach_skeleton, alpha, overlay, 1 - alpha, 0)

        # Draw player's skeleton (red)
        player_data = load_json_data('player_data.json')
        if str(p_frame) in [str(f['frame']) for f in player_data]:
            p_keypoints = next(f['player_keypoints'] for f in player_data if f['frame'] == p_frame)
            # Draw landmarks
            for idx, kp in p_keypoints.items():
                if kp['visibility'] < 0.5:  # Skip low-visibility keypoints
                    continue
                x, y = int(kp['x'] * frame_width), int(kp['y'] * frame_height)
                cv2.circle(overlay, (x, y), 5, (0, 0, 255), -1)  # Red dots
            # Draw connections
            for connection in pose_connections:
                start_idx, end_idx = connection
                if str(start_idx) in p_keypoints and str(end_idx) in p_keypoints:
                    start_kp = p_keypoints[str(start_idx)]
                    end_kp = p_keypoints[str(end_idx)]
                    if start_kp['visibility'] > 0.5 and end_kp['visibility'] > 0.5:
                        start_point = (int(start_kp['x'] * frame_width), int(start_kp['y'] * frame_height))
                        end_point = (int(end_kp['x'] * frame_width), int(end_kp['y'] * frame_height))
                        cv2.line(overlay, start_point, end_point, (0, 0, 255), 2)

        out.write(overlay)

    baseline_cap.release()
    player_cap.release()
    out.release()
    print(f"Ghost overlay video saved to {output_video}")

def generate_textual_feedback(movement_analysis):
    feedback = []
    form = movement_analysis['form_accuracy']
    timing = movement_analysis['timing_consistency']
    completion = movement_analysis['drill_completion']

    # Form feedback
    if form['left_leg_angle_diff'] > 15:
        feedback.append("Improve left leg form: Bend your knee more during turns or sprints to match the coach's posture.")
    if form['right_leg_angle_diff'] > 15:
        feedback.append("Improve right leg form: Align your ankle and knee closer to the coach's during footwork.")
    if form['overall_angle_diff'] < 15:
        feedback.append("Good overall form! Your posture closely matches the coach's.")
    elif form['overall_angle_diff'] > 25:
        feedback.append("Significant form differences detected. Focus on aligning your body posture with the coach's.")

    # Timing feedback
    if timing['avg_frame_offset'] > 15:
        feedback.append(f"Timing issue: Your actions are off by ~{int(timing['avg_frame_offset'])} frames. Try to match the coach's pace, especially during transitions.")
    if timing['dtw_distance'] > 50:
        feedback.append("Large timing differences detected. Practice maintaining consistent speed throughout the drill.")

    # Drill completion feedback
    if not completion['missing_cones']:
        feedback.append("Great job! You interacted with all cones as in the coach's drill.")
    else:
        feedback.append(f"Missed cones: {completion['missing_cones']}. Ensure you navigate all cones as shown.")
    if completion['ball_interaction']:
        feedback.append("Good ball control: You successfully interacted with the ball.")
    else:
        feedback.append("No ball interaction detected. Ensure you engage with the ball as in the coach's drill.")

    return feedback

def create_dashboard(movement_analysis, output_html):
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Football Drill Performance Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ text-align: center; }}
            .container {{ max-width: 800px; margin: auto; }}
            canvas {{ margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Football Drill Performance Dashboard</h1>
            <h2>Form Accuracy</h2>
            <canvas id="formChart"></canvas>
            <h2>Timing Consistency</h2>
            <canvas id="timingChart"></canvas>
            <h2>Drill Completion</h2>
            <p>Completed Cones: {movement_analysis['drill_completion']['completed_cones']}</p>
            <p>Missing Cones: {movement_analysis['drill_completion']['missing_cones']}</p>
            <p>Ball Interaction: {'Yes' if movement_analysis['drill_completion']['ball_interaction'] else 'No'}</p>
            <h2>Feedback</h2>
            <ul>
                {"".join(f"<li>{f}</li>" for f in generate_textual_feedback(movement_analysis))}
            </ul>
        </div>
        <script>
            const formCtx = document.getElementById('formChart').getContext('2d');
            new Chart(formCtx, {{
                type: 'bar',
                data: {{
                    labels: ['Left Leg', 'Right Leg', 'Overall'],
                    datasets: [{{
                        label: 'Angle Difference (degrees)',
                        data: [{movement_analysis['form_accuracy']['left_leg_angle_diff']}, 
                               {movement_analysis['form_accuracy']['right_leg_angle_diff']}, 
                               {movement_analysis['form_accuracy']['overall_angle_diff']}],
                        backgroundColor: ['#36A2EB', '#FF6384', '#FFCE56']
                    }}]
                }},
                options: {{ scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: 'Degrees' }} }} }} }}
            }});

            const timingCtx = document.getElementById('timingChart').getContext('2d');
            new Chart(timingCtx, {{
                type: 'bar',
                data: {{
                    labels: ['Avg Frame Offset', 'DTW Distance'],
                    datasets: [{{
                        label: 'Timing Metrics',
                        data: [{movement_analysis['timing_consistency']['avg_frame_offset']}, 
                               {movement_analysis['timing_consistency']['dtw_distance']}],
                        backgroundColor: ['#4BC0C0', '#9966FF']
                    }}]
                }},
                options: {{ scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: 'Value' }} }} }} }}
            }});
        </script>
    </body>
    </html>
    """
    with open(output_html, 'w') as f:
        f.write(html_content)
    print(f"Dashboard saved to {output_html}")

def main(baseline_video, player_video, alignment_json, movement_json, output_video, output_json, output_html):
    # Load data
    alignment_data = load_json_data(alignment_json)
    movement_analysis = load_json_data(movement_json)

    # Create ghost overlay video
    create_ghost_overlay(baseline_video, player_video, alignment_data, output_video)

    # Generate textual feedback
    feedback = generate_textual_feedback(movement_analysis)
    with open(output_json, 'w') as f:
        json.dump({'feedback': feedback}, f, indent=4)
    print(f"Textual feedback saved to {output_json}")

    # Create performance dashboard
    create_dashboard(movement_analysis, output_html)

# Example usage
baseline_video = 'benchmark-sample.mp4'
player_video = 'practise-sample1.mp4'
alignment_json = 'alignment_data.json'
movement_json = 'movement_analysis.json'
output_video = 'feedback_overlay.mp4'
output_json = 'feedback_text.json'
output_html = 'dashboard.html'
main(baseline_video, player_video, alignment_json, movement_json, output_video, output_json, output_html)