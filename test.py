import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os


images = []
classNames = []
path = './images'
filenames = os.listdir(path)

# function to load and preprocess the images
def load_images(filenames, size=(120, 120)):
    global images
    global classNames
   
    for filename in filenames:
        image = cv2.imread(f'{path}/{filename}', cv2.IMREAD_GRAYSCALE)
        classNames.append(os.path.splitext(filename)[0])
        image = cv2.resize(image, size)
        images.append(image)
    images = np.array(images)
    images = images.reshape(images.shape[0], -1)
    return images

# function to recognize faces using the trained SVM classifier
def recognize_face(face, eigen_space, mean_face, clf):
    # calculate the difference between the face and the mean face
    diff_face = face - mean_face
    
    # project the face onto the eigen space
    projected_face = np.dot(diff_face, eigen_space)
    
    # normalize the projected face to unit length
    norm = np.linalg.norm(projected_face)
    projected_face = projected_face / norm
    
    # predict the label of the projected face using the SVM classifier
    label = clf.predict([projected_face])[0]
    
    return label

# load the face images
# filenames = ["face1.jpg", "face2.jpg", "face3.jpg", "face4.jpg", "face5.jpg"]



X = load_images(filenames)

# create labels for the images
y = np.array(classNames)


# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# calculate the mean face and the eigen space using the training set
mean_face = np.mean(X_train, axis=0)
diff_faces = X_train - mean_face
cov_matrix = np.cov(diff_faces, rowvar=False)
eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix.reshape((1,1)))
idx = eigen_values.argsort()[::-1]
eigen_values = eigen_values[idx]
eigen_vectors = eigen_vectors[:, idx]
eigen_space = np.dot(eigen_vectors, diff_faces)

# save the eigen space and the mean face to files for later use
np.save("./eigen_space.npy", eigen_space)
np.save("./mean_face.npy", mean_face)

# load the eigen space and the mean face from files
eigen_space = np.load("./eigen_space.npy")
mean_face = np.load("./mean_face.npy")

# create an SVM classifier
clf = SVC(kernel="linear")

y_train = y_train.reshape(-1, 1)


print(eigen_space.shape)
print(y_train.shape)

# train the classifier on the training set
clf.fit(eigen_space, y_train)

# initialize the camera
cap = cv2.VideoCapture(0)

# loop over the frames from the camera
while True:
    # read the frame from the camera
    ret, frame = cap.read()
    
    # convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # detect faces in the frame using a Haar cascade classifier
    face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    
    # loop over the detected faces
    for (x, y, w, h) in faces:
        global face
        # extract the face from the frame
        face = gray[y:y+h, x:x+w]
        
    # resize the face to the same size as the training images
    face = cv2.resize(face, (100, 100))
    
    # recognize the face using the trained SVM classifier
    label = recognize_face(face.flatten(), eigen_space, mean_face, clf)
    
    # display the label on the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Label: {}".format(label), (x, y-10), font, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    
    # display the frame
    cv2.imshow("Frame", frame)

    # check for key presses
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key is pressed, break from the loop
    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()

