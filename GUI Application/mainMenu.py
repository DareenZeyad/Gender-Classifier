# Imports
from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
from keras.utils import get_file
import os
import numpy as np
import cv2 as cv

# Initializing
model = load_model('assets/preTrainedModel.h5')
face_cascade = cv.CascadeClassifier('assets/haarcascade_frontalface_alt.xml')

Imagefactor = 0.5
poweredImageFactor = 0.7

from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image

root = Tk()
root.geometry("800x600")
root.title('Gender Classifier')


# Functions
# Open Camera
def openCamera(event):
    global model
    global face_cascade
    genderWindow = "Gender Classification"
    # Fullscreen - Disabled
    # cv.namedWindow(genderWindow, cv.WND_PROP_FULLSCREEN)
    # cv.setWindowProperty(genderWindow, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
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
            roi_gray = gray[y: y + h, x: x + w]
            roi_color = img[y: y + h, x: x + w]
            extractedImage = np.copy(img[y: y + h, x: x + w])
            if (extractedImage.shape[0]) < 10 or (extractedImage.shape[1]) < 10:
                continue
            extractedImage = cv.resize(extractedImage, (96, 96))
            extractedImage = extractedImage.astype("float") / 255.0
            extractedImage = img_to_array(extractedImage)
            extractedImage = np.expand_dims(extractedImage, axis=0)
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
        recPts = np.array(
            [[[0, 30], [0 + widthh + 10, 30], [0 + widthh + 10, 30 + heightt + 10], [0, 30 + heightt + 10]]],
            dtype=np.int32)
        cv.fillPoly(img, recPts, (0, 0, 0))
        cv.putText(img, "Press Q to Quit", (0, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, titleColor, 2)
        cv.imshow(genderWindow, img)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break

    webcam.release()
    cv.destroyAllWindows()


# Image Input Window
def openImageWindow(event):
    global model
    path = filedialog.askopenfilename()
    img = cv.imread(path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    print(path)
    face_cascade = cv.CascadeClassifier('assets/haarcascade_frontalface_default.xml')
    currFaces = face_cascade.detectMultiScale(gray, 1.3, 5)
    faces = currFaces
    maxLength = len(faces)

    haarcascades = ['assets/haarcascade_frontalface_alt.xml', 'assets/haarcascade_frontalface_alt2.xml',
                    'assets/haarcascade_frontalface_alt_tree.xml']
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
        roi_gray = gray[y: y + h, x: x + w]
        roi_color = img[y: y + h, x: x + w]
        extractedImage = np.copy(img[y: y + h, x: x + w])
        if (extractedImage.shape[0]) < 10 or (extractedImage.shape[1]) < 10:
            continue
        extractedImage = cv.resize(extractedImage, (96, 96))
        extractedImage = extractedImage.astype("float") / 255.0
        extractedImage = img_to_array(extractedImage)
        extractedImage = np.expand_dims(extractedImage, axis=0)
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
            if widthh <= w:
                ok = True
        cv.putText(img, label, (x, y), cv.FONT_HERSHEY_SIMPLEX, fontScale, textColor, 2)
        # cv.imshow("gender detection", img)

    outputImg = img
    (imgWidth, imgHeight, imgDepth) = outputImg.shape
    outputWindow = 'Output Image'
    # Fullscreen - Disabled
    # cv.namedWindow(outputWindow, cv.WND_PROP_FULLSCREEN)
    # cv.setWindowProperty(outputWindow, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    cv.imshow(outputWindow, outputImg)
    cv.imwrite('output.jpg', outputImg)
    videoBtn.place_forget()
    videoBtn.place(x=(800 - 247) / 2, y=370)


# Loading Images
logo = Image.open("assets/Logo.png")
logo = ImageTk.PhotoImage(logo)

poweredImage = Image.open("assets/powered.png")
poweredImage = ImageTk.PhotoImage(poweredImage.resize((int(495 * poweredImageFactor), int(90 * poweredImageFactor))))

videoImage = Image.open("assets/video.png")
videoImage = ImageTk.PhotoImage(videoImage.resize((int(495 * Imagefactor), int(90 * Imagefactor))))

inputImage = Image.open("assets/cam.png")
inputImage = ImageTk.PhotoImage(inputImage.resize((int(495 * Imagefactor), int(90 * Imagefactor))))

# Logo Label
logoLbl = Label(root, image=logo)
logoLbl.place(x=(800 - 500) / 2 - 30, y=10)

poweredLabel = Label(root, image=poweredImage)
poweredLabel.place(x=(800 - int(495 * poweredImageFactor)) / 2, y=500)

# Buttons
inputImageBtn = Button(root)
inputImageBtn.config(image=inputImage)
inputImageBtn.place(x=(800 - 247) / 2, y=300)
inputImageBtn.bind('<Button-1>', openImageWindow)

videoBtn = Button(root)
videoBtn.config(image=videoImage)
videoBtn.place(x=(800 - 247) / 2, y=370)
videoBtn.bind('<Button-1>', openCamera)

# btn.place_forget()

root.mainloop()