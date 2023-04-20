import cv2
import os
import time


def train():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Get the name of the person
    name = input('Enter your name: ')

    if(not name.isalpha()):
        print("Please enter a valid name")
        exit()
    # Set up camera
    cap = cv2.VideoCapture(0)

    # Set up save directory
    save_dir = './voters/'+name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Set up image counter
    img_count = 0
    takesamples = 30

    # Capture images
    while True:
        # Read frame from camera
        ret, frame = cap.read()

        # Show frame
        frame = cv2.resize(frame, (420, 420))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Draw rectangles around the detected faces and save the images
        if(img_count<=takesamples):
            for (x, y, w, h) in faces:
                if(len(faces)<=1):
                    img_count += 1
                    img_name = name + "_" + str(img_count) + ".png"
                    cv2.imwrite(os.path.join(save_dir, img_name), frame)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    print(f'Saved {img_name}')
                    time.sleep(0.5)
        else:
            print("Done")
            break

        # If 'q' key is pressed, quit
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    # Release camera and close window
    cap.release()
    cv2.destroyAllWindows()