import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def detect_pose_and_count_arm_movements(frames):
    arm_up = False
    arm_movements_count = 0

    def is_arm_up(landmarks):
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]

        left_arm_up = left_elbow.y < left_shoulder.y
        right_arm_up = right_elbow.y < right_shoulder.y

        return left_arm_up or right_arm_up

    for frame in frames:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            if is_arm_up(results.pose_landmarks.landmark):
                if not arm_up:
                    arm_up = True
                    arm_movements_count += 1
            else:
                arm_up = False

    return arm_movements_count
