import cv2
import mediapipe as mp
import math
import numpy as np
from collections import Counter

def is_mouth_open(landmark, frame_height, frame_width):
    pt1 = [landmark[13].x * frame_width, landmark[13].y * frame_height]
    pt2 = [landmark[14].x * frame_width, landmark[14].y  * frame_height]

    dist_mouth = math.dist(pt1, pt2)
    if(dist_mouth > 10):
        return "Mouth opened"
    else:
        return "Mouth closed" 

def is_left_eye_closed(landmark, frame_height, frame_width):
    left_top = [landmark[159].x * frame_width, landmark[159].y * frame_height]
    left_bottom = [landmark[145].x * frame_width, landmark[145].y * frame_height]
    left_left = [landmark[33].x * frame_width, landmark[33].y * frame_height]
    left_right = [landmark[133].x * frame_width, landmark[133].y * frame_height]

    ver_left_dist = math.dist(left_top, left_bottom)
    hor_left_dist = math.dist(left_left, left_right)

    ratio_left = int((ver_left_dist/hor_left_dist)*100)
    if(ratio_left < 30):
        return "Left eye closed"
    else:
        return "Left eye opened"

def is_right_eye_closed(landmark, frame_height, frame_width):
    right_top = [landmark[386].x * frame_width, landmark[386].y * frame_height]
    right_bottom = [landmark[374].x * frame_width, landmark[374].y * frame_height]
    right_left = [landmark[362].x * frame_width, landmark[362].y * frame_height]
    right_right = [landmark[263].x * frame_width, landmark[263].y * frame_height]

    ver_right_dist = math.dist(right_top, right_bottom)
    hor_right_dist = math.dist(right_left, right_right)

    ratio_right = int((ver_right_dist/hor_right_dist)*100)
    if(ratio_right < 30):
        return "Left eye closed"
    else:
        return "Left eye opened" 
    
def is_anomaly(landmark, frame_height, frame_width):
    left_eye_left = [landmark[33].x * frame_width, landmark[33].y * frame_height]
    right_eye_right = [landmark[263].x * frame_width, landmark[263].y * frame_height]

    mouth_left = [landmark[291].x * frame_width, landmark[291].y * frame_height]
    mouth_right = [landmark[61].x * frame_width, landmark[61].y * frame_height]

    left_dist = math.dist(right_eye_right, mouth_left)
    right_dist = math.dist(left_eye_left, mouth_right)

    threshold = 10 
    expected_dist = 190

    if abs(left_dist - right_dist) > threshold or \
       (min(left_dist, right_dist) < expected_dist and max(left_dist, right_dist) > expected_dist + threshold):
        return "Anomalous movement"
    else:
        return "Commom movement"
    
def add_activity_if_new(activity_list, activity):
    if not activity_list or activity_list[-1] != activity:
        activity_list.append(activity)

def activity_detection(frame, frame_width, frame_height, 
    mouth_activities,
    left_eye_activities,
    right_eye_activities,
    anomaly_activities):

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()

    # frame = _frame.copy()

    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_image)

    if result.multi_face_landmarks: 
        for facial_landmarks in result.multi_face_landmarks:

            mouth_activity = is_mouth_open(facial_landmarks.landmark, frame_height, frame_width)
            left_eye_activity = is_left_eye_closed(facial_landmarks.landmark, frame_height, frame_width)
            right_eye_activity = is_right_eye_closed(facial_landmarks.landmark, frame_height, frame_width)
            anomaly_activity = is_anomaly(facial_landmarks.landmark, frame_height, frame_width)
            
            add_activity_if_new(mouth_activities, mouth_activity)
            add_activity_if_new(left_eye_activities, left_eye_activity)
            add_activity_if_new(right_eye_activities, right_eye_activity)
            add_activity_if_new(anomaly_activities, anomaly_activity)

            for i in range(0, 468):
                point = facial_landmarks.landmark[i]
                x = int(point.x * frame_width)
                y = int(point.y * frame_height)

                cv2.circle(frame, (x, y), 3, (100, 0, 0), -1)
    else:
        print("")

    concat_activities = np.concatenate((mouth_activities, left_eye_activities, right_eye_activities, anomaly_activities))
    concat_activities = Counter(concat_activities)    
    return frame, concat_activities