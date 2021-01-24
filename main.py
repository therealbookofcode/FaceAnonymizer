#create a facial recognition that blurs out faces
#import stuff 'n stuff'
import numpy as np
import cv2
import dlib

# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# get video from webcam
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    # Convert image into grayscale
    gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

    # Use detector to find landmarks
    faces = detector(gray)

    for face in faces:
        x1 = face.left()  # left point
        y1 = face.top()  # top point
        x2 = face.right()  # right point
        y2 = face.bottom()  # bottom point
        #crop the face
        imgs = frame[y1:y2, x1:x2]

        # Blur the face image
#option 1
        imgs = cv2.GaussianBlur(imgs, (99, 99), 30)
#option 2
        #color = tuple((0,0,0))
        #imgs[:] = color
#option 3
        '''(h, w) = imgs.shape[:2]
        xSteps = np.linspace(0, w, 13 + 1, dtype="int")
        ySteps = np.linspace(0, h, 13 + 1, dtype="int")
        for i in range(1, len(ySteps)):
            for j in range(1, len(xSteps)):
                startX = xSteps[j - 1]
                startY = ySteps[i - 1]
                endX = xSteps[j]
                endY = ySteps[i]
                roi = imgs[startY:endY, startX:endX]
                (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
                cv2.rectangle(imgs, (startX, startY), (endX, endY),(B, G, R), -1)'''

        # Put the blurred face region back into the frame image
        
        frame[y1:y2, x1:x2] = imgs
    # show the image
    cv2.imshow("Face", frame)
    # Exit when escape is pressed
    if cv2.waitKey(delay=1) == 27:
        break

# When everything done, release the video capture and video write objects
cap.release()

# Close all windows
cv2.destroyAllWindows()
