# cSpell:disable 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def process_video(video_path, model, n_frames):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % n_frames == 0:
            processed_frame = checking(frame, model)
            cv2.imshow('Processed Frame', processed_frame)
            output_filename = f"processedframes/frame{frame_count:04d}.jpeg"
            cv2.imwrite(output_filename, processed_frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
        frame_count += 1
    cap.release()
    cv2.destroyAllWindows()

def checking(img, model):
    label = {0: "female", 1: "male"}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = cascade.detectMultiScale(gray, 1.1, 7)

    for x, y, w, h in faces:
        face = img[y:y+h, x:x+w]
        # Check if the face image is not empty
        if face.size == 0:
            continue
        new_width = 160
        new_height = 160
        face_resized = cv2.resize(face, (new_width, new_height))

        face_scaled = face_resized / 255.0
        reshape = np.reshape(face_scaled, (1, new_height, new_width, 3))
        result = np.argmax(model.predict(reshape), axis=-1)

        if result == 0:
            cv2.rectangle(img, (x-10, y), (x+w, y+h), (255, 0, 0), 4)
            cv2.rectangle(img, (x-10, y-50), (x+w, y), (255, 0, 0), -1)
            cv2.putText(img, label[0], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        elif result == 1:
            cv2.rectangle(img, (x-10, y), (x+w, y+h), (0, 0, 255), 4)
            cv2.rectangle(img, (x-10, y-50), (x+w, y), (0, 0, 255), -1)
            cv2.putText(img, label[1], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return img

# Load your trained gender classification model
gender_model = tf.keras.models.load_model('ModeloClasificacionGenero.h5', compile=False)
process_video('videopeople.mp4', gender_model, 5)