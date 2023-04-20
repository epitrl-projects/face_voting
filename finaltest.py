import cv2
import face_recognition
import os
import numpy as np



faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
path = 'voters'
images = []
classNames = []

# Loop through each subdirectory in the root directory
for label in os.listdir(path):
    label_folder = path+"/"+label
    
    # Check if the current item in the root directory is a folder
    if os.path.isdir(label_folder):
        
        # Loop through each image file in the current folder
        for filename in os.listdir(label_folder):
            img_path = os.path.join(label_folder, filename)
            
            # Check if the current file is an image file
            if img_path.endswith(".jpg") or img_path.endswith(".png"):
                print(label_folder + "/" + filename)
                
                # Load the image and append it to the list of images
                curImg = cv2.imread(label_folder + "/" + filename)
                images.append(curImg)
                
                # Append the label (folder name) to the list of class names
                classNames.append(filename.split('.')[0])
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded_face =face_recognition.face_encodings(img)
        encodeList.append(encoded_face)
    return encodeList
encoded_face_train = findEncodings(images)



cap  = cv2.VideoCapture(0)
validity = [0,True]
nametxt = "not"
while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25,0.25)
    
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faces_in_frame = face_recognition.face_locations(imgS)
    encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)
    if nametxt == "not":
       print('not')
    try:
        for encode_face, faceloc in zip(encoded_faces,faces_in_frame):
            matches = face_recognition.compare_faces(encoded_face_train, encode_face)
            faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
            if len(faceDist) > 0:
                matchIndex = np.argmin(faceDist)
                # Rest of your code for handling the matching face
            else:
                print("No faces found")
                break
            matchIndex = np.argmin(faceDist)
            print(matches)
            
           
            if matches[matchIndex]:
                name = classNames[matchIndex].upper().lower()
                y1,x2,y2,x1 = faceloc
                # since we scaled down by 4 times
                y1, x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(img, (x1,y2-35),(x2,y2), (0,255,0), cv2.FILLED)
                cv2.putText(img,name, (x1+6,y2-5), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                if nametxt != name:
                   print("scanning...")
                if validity[0]>=20:
                    nametxt = name
                    validity[0] = 0
                    print("welcome "+name)
                break
            else:
                print("not found")
                validity[0] = 0
                y1,x2,y2,x1 = faceloc
                nametxt = "not"
                # since we scaled down by 4 times
                y1, x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(255, 0, 13),2)
                cv2.rectangle(img, (x1,y2-35),(x2,y2), (255, 0, 13), cv2.FILLED)
                cv2.putText(img,"Not found", (x1+6,y2-5), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
              
                break
    except KeyboardInterrupt:
        # If there is a KeyboardInterrupt (when you press ctrl+c), exit the program and cleanup
        print("Cleaning up!")
    validity[0]=validity[0]+1  
    cv2.imshow('webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break