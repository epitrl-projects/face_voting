import cv2
import face_recognition
import os

# Load the known faces and their names
known_faces = []
known_names = []
for filename in os.listdir('./voters'):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        # Load the image and encode the face
        img = face_recognition.load_image_file(os.path.join('./voters', filename))
        if(len(face_recognition.face_encodings(img))>0):
            encoding = face_recognition.face_encodings(img)[0]
            
            # Add the encoding and name to the lists of known faces and names
            known_faces.append(encoding)
            known_names.append(filename.split('.')[0])

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Load the face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    # If end of video is reached, break loop
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Recognize each detected face
    for (x, y, w, h) in faces:
        # Crop the face image
        face_img = frame[y:y+h, x:x+w]
        
        

        # Encode the face image
        faces_in_frame = face_recognition.face_locations(face_img)
        face_encoding = face_recognition.face_encodings(face_img, faces_in_frame)

        # Compare the face encoding to the known faces
        matches = face_recognition.compare_faces(known_faces, face_encoding)

        # Find the best match
        best_match_index = matches.index(True) if True in matches else -1
        if best_match_index != -1:
            # Predict the name of the person
            name = known_names[best_match_index]

            # Draw a label with the name below the face
            cv2.putText(frame, name, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Webcam Face Recognition', frame)

    # Exit if the user presses 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
