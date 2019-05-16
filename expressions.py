import cv2
import glob
import numpy as np

emojis = ["neutral", "anger", "happy", "sadness", "surprise", "fear"]

# Train fisher model
# print("Start training fisher model")
# fisher_face = cv2.face.FisherFaceRecognizer_create()

# training_images, training_labels = [], []
# for emoji in emojis:
#     images = glob.glob("final_dataset\\%s\\*" % emoji)
#     for image in images:
#         img = cv2.imread(image)
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         training_images.append(gray)
#         training_labels.append(emojis.index(emoji))

# fisher_face.train(training_images, np.asarray(training_labels))
# print("Model trained with %d images" % len(training_labels))
# fisher_face.save("model.xml")

fisher_face = cv2.face.FisherFaceRecognizer_create()
fisher_face.read("model.xml")

face_detectors = [cv2.CascadeClassifier("certificates/haarcascade_frontalface_default.xml"),
                  cv2.CascadeClassifier(
                      "certificates/haarcascade_frontalface_alt2.xml"),
                  cv2.CascadeClassifier(
                      "certificates/haarcascade_frontalface_alt.xml"),
                  cv2.CascadeClassifier("certificates/haarcascade_frontalface_alt_tree.xml")]

cap = cv2.VideoCapture(0)
cv2.namedWindow("Expressions")

while True:
    ret, frame = cap.read()
    if ret == False or cv2.waitKey(5) == ord('q'):
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = []
    for face_detector in face_detectors:
        tmp = face_detector.detectMultiScale(gray, scaleFactor=1.1,
                                             minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        if len(tmp) > 0:
            faces = tmp
            break

    for (x, y, w, h) in faces:
        gray2 = cv2.resize(gray[y:y+h, x:x+w], (350, 350))
        pred, conf = fisher_face.predict(gray2)
        # print(pred, conf)
        cv2.rectangle(frame, (x, y), (x+w, y + h), (0, 200, 0), 2)
        cv2.rectangle(frame, (x, y - 30), (x+w, y), (0, 200, 0), -1)
        cv2.putText(frame, emojis[pred], (x, y - 10),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255))
    cv2.imshow("Expressions", frame)

cap.release()
cv2.destroyAllWindows()
