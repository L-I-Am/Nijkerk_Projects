import cv2
import time

counter = 0
previousAmountFaces = 0

camera = cv2.VideoCapture(0)
rval = True

# keep looping
while rval:
    # grab the current frame
    rval, frame = camera.read()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # load in haar cascades for eyes and faces
    haar_face_cascade = cv2.CascadeClassifier(r'..\Python3\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
    haar_eye_cascade = cv2.CascadeClassifier(r'..\Python3\Lib\site-packages\cv2\data\haarcascade_eye.xml')

    # detect eyes and faces on the grayscale image
    faces = haar_face_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=14)
    eyes = haar_eye_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=14)

    if previousAmountFaces != len(faces):
        if previousAmountFaces < len(faces):
            counter = counter + (len(faces) - previousAmountFaces)
        previousAmountFaces = len(faces)

    # draw rectangles around the faces and eyes
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Put text on frame

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 465)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2

    cv2.putText(frame, counter.__str__(),
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)

    # show image
    cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Frame", frame)

    # if the 'q' or escape key is pressed, stop the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break

camera.release()
cv2.destroyAllWindows()
