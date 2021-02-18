# Step 5 - Detect Input Camera Frames
from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
from keras.utils import get_file
import os
import numpy as np
import cv2 as cv

model = load_model('preTrainedModel.h5')
genderWindow = "Gender Classification"
cv.namedWindow(genderWindow, cv.WND_PROP_FULLSCREEN)
cv.setWindowProperty(genderWindow, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
webcam = cv.VideoCapture(0)
while webcam.isOpened():
    status, frame = webcam.read()
    if not status:
        print("Could not read frame")
        exit()
    img = frame
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        rectangleColor = (0, 255, 0)
        cv.rectangle(img, (x, y), (x + w, y + h), rectangleColor, 2)
        extractedImage = np.copy(img[y : y + h, x : x + w])
        if (extractedImage.shape[0]) < 10 or (extractedImage.shape[1]) < 10:
            continue
        extractedImage = cv.resize(extractedImage, (96, 96))
        extractedImage = extractedImage.astype("float") / 255.0
        extractedImage = img_to_array(extractedImage)
        extractedImage = np.expand_dims(extractedImage, axis = 0)
        pred = model.predict(extractedImage)
        if pred[0][0] > 0.5:
            label = "Male"
            textColor = (255, 0, 0)
        else:
            label = "Female"
            textColor = (0, 0, 255)
        cv.putText(img, label, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.7, textColor, 2)
    titleColor = (0, 0, 255)
    (widthh, heightt), baseline = cv.getTextSize("Press Q to Quit", cv.FONT_HERSHEY_SIMPLEX, 0.7, 2)
#    cv.rectangle(img, (0, 30), (0 + widthh + 10, 30 + heightt + 10), (0, 0, 0), 2)
    recPts = np.array( [[[0, 30], [0 + widthh + 10, 30], [0 + widthh + 10, 30 + heightt + 10], [0, 30 + heightt + 10]]], dtype=np.int32 )
    cv.fillPoly(img, recPts, (0, 0, 0))
    cv.putText(img, "Press Q to Quit", (0, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, titleColor, 2)
    cv.imshow(genderWindow, img)
    if cv.waitKey(10) & 0xFF == ord('q'):
        break

webcam.release()
cv.destroyAllWindows()