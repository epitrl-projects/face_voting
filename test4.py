import os
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import train

import re
import csv
import gui

statevars = ["T", "R"]

# Get the name of the person
name = input('Enter Procedure: ')

if (not name.isalpha() and name not in statevars):
    print("Please enter a valid Procedure")
    exit()


if (name == "T"):
    train.train()

datalabels={}
def prepare_dataset(path):
    count = 0
    labels, faces = [], []
    for person in os.listdir(path):
        person_path = os.path.join(path, person)
        if os.path.isdir(person_path):
            for img in os.listdir(person_path):
                img_path = os.path.join(person_path, img)
                face = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                face = cv2.resize(face, (100, 100))

                faces.append(face)
                count = count + 1
                datalabels[count] = img.split('.')[0]
                labels.append(count)
    return np.array(faces), np.array(labels)

def train_models(faces, labels):
    lbph = cv2.face.LBPHFaceRecognizer_create()
    eigen = cv2.face.EigenFaceRecognizer_create()
    lbph.train(faces, labels)
    eigen.train(faces, labels)
    return lbph, eigen

def save_models(lbph, eigen):
    lbph.write('lbph_model.yml')
    eigen.write('eigen_model.yml')

def load_models():
    lbph = cv2.face.LBPHFaceRecognizer_create()
    eigen = cv2.face.EigenFaceRecognizer_create()
    lbph.read('lbph_model.yml')
    eigen.read('eigen_model.yml')
    return lbph, eigen

def evaluate_models(lbph, eigen, faces, labels):
    lbph_preds, eigen_preds = [], []
    for face in faces:
        _1, lbph_pred = lbph.predict(face)
        _, eigen_pred = eigen.predict(face)
        cv2.putText(face, str(_1), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(face, str(_), (80, 50), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow('face', face)
        lbph_preds.append(_1)
        eigen_preds.append(_)
        print(_1,_, lbph_pred, eigen_pred)

    lbph_accuracy = accuracy_score(labels, lbph_preds)
    lbph_precision = precision_score(labels, lbph_preds, average='weighted')
    lbph_recall = recall_score(labels, lbph_preds, average='weighted')
    lbph_f1 = f1_score(labels, lbph_preds, average='weighted')

    eigen_accuracy = accuracy_score(labels, eigen_preds)
    eigen_precision = precision_score(labels, eigen_preds, average='weighted')
    eigen_recall = recall_score(labels, eigen_preds, average='weighted')
    eigen_f1 = f1_score(labels, eigen_preds, average='weighted')
    
    return lbph_accuracy, lbph_precision, lbph_recall, lbph_f1,eigen_accuracy,eigen_precision,eigen_recall,eigen_f1



def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=2, colour=(0, 255, 0)):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y-size[1]), (x+size[0], y), colour, cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale,
                (255, 255, 255), thickness)


def main():
    confirmstate = "nan"
    confirmstatenum = 0
    dataset_path = './voters'
    dataset_path = './test'
    faces, labels = prepare_dataset(dataset_path)
    testfaces, testlabels = prepare_dataset(dataset_path)
    lbph, eigen = train_models(faces, labels)
    save_models(lbph, eigen)
    lbph, eigen = load_models()


    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    counting = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if(counting<=3):
            try:
                lbph_accuracy, lbph_precision, lbph_recall, lbph_f1,eigen_accuracy,eigen_precision,eigen_recall,eigen_f1 = evaluate_models(lbph, eigen, testfaces, testlabels)
            except:
                print("evaluate_models issue ")
            counting = counting + 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_gray = gray[y:y+h, x:x+w]
            face_gray_resized = cv2.resize(face_gray, (100, 100))

            _1, lbph_pred = lbph.predict(face_gray_resized)
            _, eigen_pred = eigen.predict(face_gray_resized)
            lbph_label = datalabels[_1]
            eigen_label = datalabels[_]

            print(lbph_pred, eigen_pred) #120 3500
            label_text = ""

            if(lbph_pred<=120):
                label_text = lbph_label
                cv2.putText(frame, f'{lbph_label}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif(eigen_pred<=3500):
                label_text = eigen_label
                cv2.putText(frame, f'{eigen_label}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f'labels : {_1}, {_}', (x, y-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # lbph_accuracy, lbph_precision, lbph_recall, lbph_f1,eigen_accuracy,eigen_precision,eigen_recall,eigen_f1

            label_text = re.sub(r'[\d_]+', '', label_text)
            # print(label_text)

            if (confirmstatenum >= 40):
                        confirmstate = label_text
                        # print(confirmstate)
                        gui.callGUI(label_text)
                        while True:
                            if (gui.windowopenstate == 1):
                                print("Waiting for person vote")
                            else:
                                break
                        confirmstatenum = 0
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


        cv2.putText(frame, f'lbph Accuracy: {lbph_accuracy:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'lbph precision: {lbph_precision:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'lbph recall: {lbph_recall:.2f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'lbph F1 Score: {lbph_f1:.2f}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame, f'eigen Accuracy: {eigen_accuracy:.2f}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'eigen precision: {eigen_precision:.2f}', (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'eigen recall: {eigen_recall:.2f}', (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'eigen F1 Score: {eigen_f1:.2f}', (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
