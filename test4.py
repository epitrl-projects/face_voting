import cv2
import os
import numpy as np
import csv
import gui
import train
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



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


# Load the known faces and their labels
known_faces = []
known_labels = []
true_labels = []


# Load training images and labels
folder_path = "./voters"

def harcascade_model_load_images_from_folder(folder_path):
    for label in os.listdir(folder_path):
        label_folder = os.path.join(folder_path, label)
        if os.path.isdir(label_folder):
            for filename in os.listdir(label_folder):
                img_path = os.path.join(label_folder, filename)
                if img_path.endswith(".jpg") or img_path.endswith(".png"):
                    img = cv2.imread(os.path.join(label_folder, filename))
                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                    faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)
                    if len(faces) > 0:
                        (x, y, w, h) = faces[0]
                        encoding = cv2.resize(img_gray[y:y+h, x:x+w], (480, 480))
                        known_faces.append(encoding)
                        known_labels.append(filename.split('.')[0])
    return known_faces

harcascade_imgs=harcascade_model_load_images_from_folder(folder_path)


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
lbph_recognizer = cv2.face.LBPHFaceRecognizer_create()
lbph_recognizer.train(images, labels)

face_recognizer.write('eigenface_model.xml')
lbph_recognizer.write('lpfh_model.xml')


face_recognizer.read('eigenface_model.xml')
lbph_recognizer.read('lpfh_model.xml')


# Initialize variables for calculating metrics and storing predicted labels
lbph_pred_labels = []
eigen_pred_labels = []

# Open the video capture device
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Read a frame from the video capture device
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame using the Haar Cascade classifier
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Loop over each detected face
    for (x, y, w, h) in faces:
        # Extract the face region from the grayscale frame
        face_gray = gray[y:y+h, x:x+w]

        # Resize the face region to the required size for the face recognition models
        face_resized = cv2.resize(face_gray, (420, 420))

        # Recognize the face using the LBPH model
        lbph_label, lbph_confidence = lbph_recognizer.predict(face_resized)
        label1_text = get_key(lbph_label, label_dict)
        label1_text = re.sub(r'[\d_]+', '', label1_text)
        lbph_pred_labels.append(lbph_label)

        # Recognize the face using the Eigenfaces model face_recognizer
        eigen_label, eigen_confidence = face_recognizer.predict(face_resized)
        label_text = get_key(eigen_label, label_dict)
        label_text = re.sub(r'[\d_]+', '', label_text)
        eigen_pred_labels.append(eigen_label)

        # Draw a rectangle around the detected face and display the predicted label and confidence
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'LBPH: {lbph_label} ({lbph_confidence:.2f})', (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f'Eigenfaces: {eigen_label} ({eigen_confidence:.2f})', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Add the true label for the detected face to the list of true labels
        true_labels.append(lbph_label)  # Replace 0 with the true label for the detected face

    # Display the frame
    cv2.imshow('Face Recognition', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Make sure the input variables have the same number of samples
    if len(true_labels) == len(lbph_pred_labels):
        # Calculate the accuracy, precision, recall, and F1 score for the LBPH model
        lbph_accuracy = accuracy_score(true_labels, lbph_pred_labels)
        lbph_precision = precision_score(true_labels, lbph_pred_labels, average='weighted')
        lbph_recall = recall_score(true_labels, lbph_pred_labels, average='weighted')
        lbph_f1_score = f1_score(true_labels, lbph_pred_labels, average='weighted')

        # Calculate the accuracy, precision, recall, and F1 score for the Eigenfaces model
        eigen_accuracy = accuracy_score(true_labels, eigen_pred_labels)
        eigen_precision = precision_score(true_labels, eigen_pred_labels, average='weighted')
        eigen_recall = recall_score(true_labels, eigen_pred_labels, average='weighted')
        eigen_f1_score = f1_score(true_labels, eigen_pred_labels, average='weighted')

        # Print the metrics for both models and select the best output label
        if lbph_f1_score > eigen_f1_score:
            print('LBPH model is better')
            best_labels = lbph_pred_labels
        else:
            print('Eigenfaces model is better')
            best_labels = eigen_pred_labels

        print(f'LBPH accuracy: {lbph_accuracy:.2f}')
        print(f'LBPH precision: {lbph_precision:.2f}')
        print(f'LBPH recall: {lbph_recall:.2f}')
        print(f'LBPH F1 score: {lbph_f1_score:.2f}')

        print(f'Eigenfaces accuracy: {eigen_accuracy:.2f}')
        print(f'Eigenfaces precision: {eigen_precision:.2f}')
        print(f'Eigenfaces recall: {eigen_recall:.2f}')
        print(f'Eigenfaces F1 score: {eigen_f1_score:.2f}')

        # Print all predicted labels and confidence values
        for i in range(len(best_labels)):
            print(f'Frame {i+1}: LBPH: {lbph_pred_labels[i]} ({lbph_confidence:.2f}), Eigenfaces: {eigen_pred_labels[i]} ({eigen_confidence:.2f})')

        # Do something with the best output labels
        # print(best_labels)

# Release the video capture device and close all windows
cap.release()
cv2.destroyAllWindows()
