import cv2
import numpy as np

classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier("haarcascade_eye.xml")

camera = cv2.VideoCapture(0)
sample = 1
sample_numbers = 25

width, height = 220, 220

id = input("Digite seu identificador: ")

print("Capturando as faces...")

while (True):
    connected, image = camera.read()
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    facesDetectadas = classifier.detectMultiScale(gray_image,
                                                     scaleFactor=1.5,
                                                     minSize=(150, 150))

    for (x, y, l, a) in facesDetectadas:
        cv2.rectangle(image, (x, y), (x + l, y + a), (0, 0, 255), 2)
        region = image[y:y + a, x:x + l]
        gray_eye_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        detected_eyes = eye_classifier.detectMultiScale(gray_eye_region)

        for(ox, oy, ol, oa) in detected_eyes:
            cv2.rectangle(region, (ox, oy), (ox + ol, oy + oa), (0, 255, 0), 2)

            if cv2.waitKey(1) & 0XFF == ord("q"):
                if np.average(gray_image) > 110:
                    face_image = cv2.resize(
                        gray_image[y:y + a, x:x + l], (width, height))
                    cv2.imwrite("photos/" + str(id) +
                                "-" + str(sample) + ".jpg", face_image)
                    print("Foto " + str(sample) + " capturada com sucesso!")
                    sample += 1
                    
    cv2.imshow("Face", image)
    cv2.waitKey(1)

    if(sample >= sample_numbers + 1):
        break

print("Faces capturadas com sucesso!")

camera.release()
cv2.destroyAllWindows()
