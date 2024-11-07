import cv2
import os
from FaceDetection import detect_faces_in_frame
from EmotionAnalysis import analyze_emotions_in_face
from ActivityDetection import detect_pose_and_count_arm_movements
from SummaryGeneration import generate_summary

def analyze_video(video_filename="video.mp4", output_filename="output_video.mp4", summary_filename="summary.txt"):
    video_path = os.path.join(os.getcwd(), video_filename)
    summary_path = os.path.join(os.getcwd(), summary_filename)

    if not os.path.isfile(video_path):
        print(f"Erro: Arquivo de vídeo '{video_filename}' não encontrado na raiz do projeto.")
        return
    
    cap = cv2.VideoCapture(video_path)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    fps = cap.get(cv2.CAP_PROP_FPS)          
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    activities = []
    emotions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (frame_width // 2, frame_height // 2))

        if frame_count % 10 == 0:
            frame, face_locations = detect_faces_in_frame(frame)
            
            for face_location in face_locations:
                emotion = analyze_emotions_in_face(frame, face_location)
                emotions.append(emotion)
                top, right, bottom, left = face_location
                cv2.putText(frame, emotion, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (255, 255, 255), 2)

            activity = detect_pose_and_count_arm_movements(frame)
            activities.append(activity)
            cv2.putText(frame, f'Cotovelo acima dos ombros: {activity}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 255, 0), 2)

        out.write(frame)

        cv2.imshow('Video Analysis', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    summary = generate_summary(activities, emotions)
    with open(summary_path, 'w') as f:
        f.write("Resumo do Vídeo:\n")
        f.write(summary)
    print("Resumo salvo em", summary_filename)

video_filename = "video.mp4"
analyze_video(video_filename)