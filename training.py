import cv2
import numpy as np
import os

eigenface = cv2.face.EigenFaceRecognizer_create()
fisherface = cv2.face.FisherFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create()


def getImageWithId():
    imagePaths = [os.path.join("photos", f) for f in os.listdir("photos")]
    faces = []
    ids = []

    for imagePath in imagePaths:
        imageFace = cv2.cvtColor(cv2.imread(
            imagePath), cv2.COLOR_BGR2GRAY)

        id = int(os.path.split(imagePath)[-1].split("-")[0])
        ids.append(id)
        faces.append(imageFace)

        cv2.imshow("Face", imageFace)
        cv2.waitKey(10)
    
    return np.array(ids), faces


ids, faces = getImageWithId()

print("Treinando...")

eigenface.train(faces, ids)
eigenface.write("classifier_eigen.yml")

lbph.train(faces, ids)
lbph.write("classifier_lpbh.yml")

print("Treinamento finalizado!")
