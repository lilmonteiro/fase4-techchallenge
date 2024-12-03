import cv2
import os
from FaceDetection import detect_faces_in_frame
from EmotionAnalysis import analyze_emotions_in_face
from ActivityDetection import activity_detection
from SummaryGeneration import generate_summary
import numpy as np

def analyze_video(video_filename="video.mp4", summary_filename="summary.txt"):
    video_path = os.path.join(os.getcwd(), video_filename)
    summary_path = os.path.join(os.getcwd(), summary_filename)

    if not os.path.isfile(video_path):
        print(f"Erro: Arquivo de vídeo '{video_filename}' não encontrado na raiz do projeto.")
        return
    
    cap = cv2.VideoCapture(video_path)
           
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_count = 0
    activities = []
    emotions = []
    mouth_activities = []
    left_eye_activities = []
    right_eye_activities = []
    anomaly_activities = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % 10 == 0:
            frame, face_locations = detect_faces_in_frame(frame)

            for face_location in face_locations:
                emotion = analyze_emotions_in_face(frame, face_location)
                emotions.append(emotion)
                top, right, bottom, left = face_location
                cv2.putText(frame, emotion, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (255, 255, 255), 2)
        
            frame, activities = activity_detection(frame, frame_width, frame_height, mouth_activities, left_eye_activities, right_eye_activities, anomaly_activities)


        cv2.imshow('Video Analysis', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1
    print(activities)
    cap.release()
    cv2.destroyAllWindows()

    summary = generate_summary(activities, emotions)
    with open(summary_path, 'w') as f:
        f.write("Resumo do Vídeo:\n")
        f.write(summary)
        f.write(f"The video contains a total of {total_frames} frames. ")
    print("Resumo salvo em", summary_filename)

if __name__ == "__main__":
    video_filename = "video.mp4"
    summary_filename = "summary.txt"

    analyze_video(video_filename, summary_filename)