from deepface import DeepFace

def analyze_emotions_in_face(frame, face_location):
    top, right, bottom, left = face_location
    face = frame[top:bottom, left:right]

    result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)

    if isinstance(result, list):
        result = result[0] 

    emotion = result.get('dominant_emotion', 'undefined')  
    return emotion