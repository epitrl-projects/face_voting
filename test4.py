import cv2
import os
import numpy as np
import csv
import gui
import train
import re


statevars = ["T", "R"]

# Get the name of the person
name = input('Enter Procedure: ')

if (not name.isalpha() and name not in statevars):
    print("Please enter a valid Procedure")
    exit()


if (name == "T"):
    train.train()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
labels = []
encoded_face_train = []
images = []
# Function to load images from a folder


def load_images_from_folder(folder_path):
    for label in os.listdir(folder_path):
        label_folder = os.path.join(folder_path, label)
        if os.path.isdir(label_folder):
            for filename in os.listdir(label_folder):
                img_path = os.path.join(label_folder, filename)
                if img_path.endswith(".jpg") or img_path.endswith(".png"):
                    img = cv2.imread(os.path.join(label_folder, filename))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    labels.append(filename.split('.')[0])
                    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    if img is not None:
                        images.append(img)
    return images


# Load training images and labels
folder_path = "./voters"
images = load_images_from_folder(folder_path)

images = [cv2.resize(img, (420, 420)) for img in images]

# Create a dictionary to map each label to an integer
label_dict = {}
for i, label in enumerate(set(labels)):
    label_dict[label] = i


def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key

    return "Value not found in dictionary"


# Convert the labels to an array of integers
labels = np.array([label_dict[label] for label in labels])

# Create face recognizer and train it
face_recognizer = cv2.face.EigenFaceRecognizer_create()
face_recognizer.train(images, labels)

face_recognizer.write('eigenface_model.xml')


face_recognizer.read('./eigenface_model.xml')


# Print training output
for i in range(len(images)):
    label, confidence = face_recognizer.predict(images[i])
    print("Image: {}, Label: {}, Confidence: {}".format(
        i+1, get_key(label, label_dict), confidence))

# Function to draw a rectangle around detected face and label it


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=2, colour=(0, 255, 0)):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y-size[1]), (x+size[0], y), colour, cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale,
                (255, 255, 255), thickness)


# Load video file for testing
cap = cv2.VideoCapture(0)


confirmstate = "nan"
confirmstatenum = 0


while True:
    ret, frame = cap.read()
    img = frame.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # If end of video is reached, break loop
    if not ret:
        break

    frame = cv2.resize(frame, (420, 420))
    # Convert frame to grayscale and detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5)

    # Only process the first detected face
    if len(faces) > 0:
        if (len(faces) > 1):
            print("More than one face detected")
            draw_label(frame, (50, 50),
                       "More than one face detected", colour=(0, 0, 255))

        x, y, w, h = faces[0]
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(gray, (420, 420))

        label, confidence = face_recognizer.predict(roi_gray)
        label_text = get_key(label, label_dict)
        label_text = re.sub(r'[\d_]+', '', label_text)
        face_size = w   # or use other facial landmarks to estimate the face size
        # print(label_text)

        # Step 4: Compute the distance
        real_world_size = 0.15  # assume the face is 15cm wide in real life
        focal_length = 500     # assume a focal length of 500 pixels
        distance = (real_world_size * focal_length) / face_size

        if (distance <= 0.35):
            if (confirmstatenum >= 40):
                confirmstate = label_text
                draw_label(frame, (x, y), label_text)
                print(confirmstate)
                gui.callGUI(label_text)
                while True:
                    if (gui.windowopenstate == 1):
                        print("Waiting for person vote")
                    else:
                        break
            else:

                with open('votes.csv', 'r', newline='') as csvfile:
                    # Create a writer object
                    csvreader = csv.reader(csvfile)
                    # Iterate over each row in the CSV file
                    data = [row[1] for row in csvreader]
                    # Check if an item exists in the list
                    print(data)
                    if label_text in data:
                        print("Already voted")
                        draw_label(frame, (x, y), label_text +
                                   " Already voted", colour=(0, 0, 255))
                        confirmstatenum = 0
                    else:
                        print('Security check......')
                        draw_label(frame, (x, y), label_text)
                        confirmstatenum = confirmstatenum + 1

        else:

            draw_label(frame, (x, y), "Move forward")

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Show frame with detected faces
    cv2.imshow('Video', frame)

    # Break loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
