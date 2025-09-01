Football Training Video Analysis System
Overview
This project is an AI-powered system to analyze football training videos, specifically for a cone dribbling drill. It compares a player's performance (practise-sample1.mp4) against a coach's baseline video (benchmark-sample.mp4) to provide actionable feedback. The system extracts skeletal keypoints, tracks objects (ball, cones), aligns videos temporally, analyzes movement patterns, and generates feedback through visual overlays, textual suggestions, and a performance dashboard.
The system is implemented in four Python scripts:

step_1.py: Processes videos to extract keypoints and track objects.
step_2(temporal_alignment).py: Aligns coach and player videos using Dynamic Time Warping (DTW).
movement_analysis.py: Analyzes movement patterns and computes performance metrics.
step_4(feedback).py: Generates visual overlays, textual feedback, and a dashboard.

Requirements

Python Version: 3.12
Dependencies: Install via pip install -r requirements.txt:mediapipe
opencv-python
numpy
dtw-python
ultralytics
filterpy


SORT Library: Clone from https://github.com/abewley/sort and place in the project directory.
Input Files:
benchmark-sample.mp4: Coach’s baseline video.
practise-sample1.mp4: Player’s practice video.


Hardware: A system with a GPU is recommended for faster video processing (MediaPipe, YOLO).

Setup

Create Virtual Environment:python3.12 -m venv env
source env/bin/activate


Install Dependencies:pip install mediapipe opencv-python numpy dtw-python ultralytics filterpy


Clone SORT:git clone https://github.com/abewley/sort


Place Videos:
Ensure benchmark-sample.mp4 and practise-sample1.mp4 are in the project directory (/home/dire/Desktop/Drill Analysis).



Scripts and Usage
1. step_1.py: Video Processing

Purpose: Extracts skeletal keypoints (MediaPipe Pose) and tracks ball/cones (YOLO with SORT) from input videos.
Inputs:
benchmark-sample.mp4
practise-sample1.mp4


Outputs:
baseline_data.json: Keypoints and object tracks for the coach’s video.
player_data.json: Keypoints and object tracks for the player’s video.
baseline_with_detections.mp4: Annotated coach video with keypoints and bounding boxes.
player_with_detections.mp4: Annotated player video.


Run:python3 step_1.py


Ensure input videos are in the directory. Modify script paths if filenames differ.


Notes: Processes videos frame-by-frame, detecting 33 MediaPipe keypoints and tracking objects with unique IDs.

2. step_2(temporal_alignment).py: Temporal Alignment

Purpose: Aligns coach and player videos using DTW to match corresponding actions (e.g., cone turns) despite speed differences.
Inputs:
baseline_data.json
player_data.json


Outputs:
alignment_data.json: Frame mappings (e.g., [{"baseline_frame": 0, "player_frame": 0}, ...]) and DTW distance.
alignment_path.png: Plot of the alignment path.


Run:python3 step_2\(temporal_alignment\).py


Or rename to step_2_temporal_alignment.py to avoid escaping parentheses:mv step_2\(temporal_alignment\).py step_2_temporal_alignment.py
python3 step_2_temporal_alignment.py




Notes: Uses keypoints (left/right hip, knee, ankle) for alignment. High DTW distance (>50) may indicate noisy keypoints or dissimilar drills.

3. movement_analysis.py: Movement Analysis

Purpose: Compares aligned keypoints to compute performance metrics:
Form Accuracy: Joint angle differences (left/right leg, overall).
Timing Consistency: Frame offsets and DTW distance.
Drill Completion: Cone and ball interactions.


Inputs:
baseline_data.json
player_data.json
alignment_data.json


Outputs:
movement_analysis.json: Metrics (e.g., {"form_accuracy": {"left_leg_angle_diff": 19.99, ...}, ...}).
angle_differences.png: Plot of angle differences over time.


Run:python3 movement_analysis.py


Notes: Focuses on hip-knee-ankle angles. Angle differences <15° are good, 15-25° suggest improvement, >25° indicate issues.

4. step_4(feedback).py: Feedback System

Purpose: Generates feedback based on analysis:
Visual overlay: Ghost comparison video (coach’s blue skeleton, player’s red skeleton).
Textual feedback: Suggestions (e.g., “Improve left leg form”).
Performance dashboard: HTML with charts for metrics.


Inputs:
benchmark-sample.mp4
practise-sample1.mp4
alignment_data.json
movement_analysis.json


Outputs:
feedback_overlay.mp4: Video with overlaid skeletons.
feedback_text.json: Feedback messages (e.g., ["Improve left leg form...", ...]).
dashboard.html: Web dashboard with bar charts and feedback.


Run:python3 step_4\(feedback\).py


Or rename to step_4_feedback.py:mv step_4\(feedback\).py step_4_feedback.py
python3 step_4_feedback.py




Notes: Open dashboard.html in a browser to view charts. Check feedback_overlay.mp4 for skeleton alignment.

Example Results
From movement_analysis.json:

Form Accuracy:
Left leg angle difference: ~20° (moderate, needs improvement).
Right leg angle difference: ~15° (good).
Overall angle difference: ~17.5° (acceptable).


Timing Consistency:
Average frame offset: 27 frames (0.9s at 30 fps, significant timing issue).
DTW distance: ~32.34 (moderate alignment).


Drill Completion:
Completed all 6 cones and interacted with the ball.


Feedback (from feedback_text.json):
Improve left leg form (knee bend).
Address timing issues (match coach’s pace).
Confirmed all cones and ball interaction.



Usage Instructions

Prepare Environment:
Set up the virtual environment and install dependencies as above.
Place benchmark-sample.mp4 and practise-sample1.mp4 in the project directory.


Run Scripts Sequentially:python3 step_1.py
python3 step_2\(temporal_alignment\).py
python3 movement_analysis.py
python3 step_4\(feedback\).py


Review Outputs:
Check JSON files for data integrity.
Watch feedback_overlay.mp4 to visualize form differences.
Open dashboard.html in a browser to view metrics and feedback.


Troubleshooting:
Ensure input videos match JSON data (frame counts).
If skeletons misalign in feedback_overlay.mp4, check alignment_path.png for DTW issues.
For high angle differences (>25°) or frame offsets (>15), verify keypoint quality or video similarity.



Notes

Performance: For long videos, consider downsampling frames to reduce processing time.
Camera Angles: Normalized keypoints help, but different camera angles may require spatial alignment.
Future Enhancements:
Segment drills into phases (sprint, turn) for granular feedback.
Add multi-session tracking to the dashboard.
Include heatmaps for keypoint deviations.



Contact
For support or customization, contact Pratham at prathamchawla21@gmail.com.