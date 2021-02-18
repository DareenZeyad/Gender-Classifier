# Step 6 - Detect An input image
# Output image is at -> output.jpg
from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
from keras.utils import get_file
import os
import numpy as np
import cv2 as cv

model = load_model('preTrainedModel.h5')


path = "predict.jpg"
img = cv.imread(path)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print(path)
face_cascade = cv.CascadeClassifier('assets/haarcascade_frontalface_default.xml')
currFaces = face_cascade.detectMultiScale(gray, 1.3, 5)
faces = currFaces
maxLength = len(faces)

haarcascades = ['assets/haarcascade_frontalface_alt.xml', 'assets/haarcascade_frontalface_alt2.xml', 'assets/haarcascade_frontalface_alt_tree.xml']
for i in range(0, 3):
    face_cascade = cv.CascadeClassifier(haarcascades[i])
    currFaces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(currFaces) > maxLength:
        faces = currFaces
        maxLength = len(currFaces)
out = []
for (x, y, w, h) in faces:
    rectangleColor = (255, 0, 0)
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
        label = "Male : "
        percentage = pred[0][0] * 100.0
        label += str(round(percentage, 2))
        label += "%"
        textColor = (0, 255, 0)
    else:
        label = "Female : "
        percentage = pred[0][1] * 100.0
        label += str(round(percentage, 2))
        label += "%"
        textColor = (255, 255, 0)
    ok = False
    fontScale = 2.1
    while ok == False:
        fontScale -= 0.1
        if fontScale <= 0.5:
            break
        (widthh, heightt), baseline = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, fontScale, 2)
        if widthh <= w :
            ok = True
    cv.putText(img, label, (x, y), cv.FONT_HERSHEY_SIMPLEX, fontScale, textColor, 2)
    # cv.imshow("gender detection", img)

cv.imwrite('output.jpg', img)