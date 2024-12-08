import cv2
import numpy as np
import traceback
import os
import albumentations as aug

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()

dataset_path = 'images'

model_path = 'face_model.yml'

target_size = (100,100)

faces = []
labels = []
names = []
label_id = 0



for person_name in os.listdir(dataset_path):
    print (os.listdir(dataset_path))
    person_path = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_path):
        continue

    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        faces.append(image)
        labels.append(label_id)
        names.append(person_name)
    
    label_id += 1

labelsfin = np.array(labels, dtype=np.int32)

print(names)

recognizer.train(faces, labelsfin)

recognizer.save(model_path)

recognizer.read(model_path)

try:
    print("Starting video capture...")

    # Open video capture
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise Exception("Unable to access the webcam.")
    
    cam.set(cv2.CAP_PROP_FPS, 10)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 3)

    # Debugging: Check if the camera is opened
    print("Camera opened successfully.")

    while True:
        try:
            ret, frame = cam.read()
            if not ret:
                print("Failed to grab frame. Check your webcam connection.")
                break

            # Convert frame to grayscale
            grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in frame
            faces = faceCascade.detectMultiScale(
                grey_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            # Debugging: Show the number of faces detected
            print(f"Faces detected: {len(faces)}")

            # Draw rectangles around detected faces
            for (x, y, w, h) in faces:
                print(f"Face found at: x={x}, y={y}, w={w}, h={h}")
                cv2.rectangle(frame, (x, y), (x + w, y + h), (250, 0, 0), 2)
                labels, confidence = 
                if confidence >= 60:
                    cv2.putText(frame, f"{names[label]}, {confidence}",(x, y -10), cv2.FONT_HERSHEY_COMPLEX, 0.9, (36,255,12),2)
                if confidence < 60:
                    cv2.putText(frame, f"Unknown",(x, y -10), cv2.FONT_HERSHEY_COMPLEX, 0.9, (36,255,12),2)

            # Display the video feed
            cv2.imshow('Identity', frame)

            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting...")
                break
        except Exception as e:
            print(f"Error inside loop: {e}")
            traceback.print_exc()  # Print traceback
            break  # Exit the loop in case of error

    # Release resources
    cam.release()
    cv2.destroyAllWindows()

except Exception as e:
    print(f"An error occurred: {e}")
    traceback.print_exc()  # Print full traceback for the error
    input("Press Enter to exit.")  # Keeps the window open