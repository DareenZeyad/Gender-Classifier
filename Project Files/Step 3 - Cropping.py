# Step 3 - Cropping Faces from the Dataset
import os
import numpy as np
import cv2 as cv
cntr = 1
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt2.xml')

allFiles = os.listdir('.')

for filename in allFiles:
    if filename.endswith('.jpg'):
        img = cv.imread(filename)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            extractedImage = np.copy(img[y:y+h, x:x+w])
            name = 'image' + str(cntr) + '.jpg'
            print(name)
            cv.imwrite(name, extractedImage)
            roi_gray = gray[y : y + h, x : x + w]
            roi_color = img[y : y + h, x : x + w]
            cntr += 1
            print(cntr)