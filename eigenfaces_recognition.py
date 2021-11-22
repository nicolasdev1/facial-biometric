import cv2

face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.EigenFaceRecognizer_create()
recognizer.read("classifier_eigen.yml")
width, height = 220, 220
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
camera = cv2.VideoCapture(0)

while (True):
    connected, image = camera.read()
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detected_faces = face_detector.detectMultiScale(gray_image,
                                                    scaleFactor=1.5,
                                                    minSize=(50, 50))

    for (x, y, l, a) in detected_faces:
        imageFace = cv2.resize(
            gray_image[y:y + a, x:x + l], (width, height))

        cv2.rectangle(image, (x, y), (x + l, y + a), (0, 0, 255), 2)
        id, confidence = recognizer.predict(imageFace)
        cv2.putText(image, str(id), (x, y + (a+30)), font, 2, (0, 0, 255))

    cv2.imshow("Face", image)
    cv2.waitKey(1)
