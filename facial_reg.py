import cv2
import pickle
video_cap = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier("cascade/haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("train.yml")
labels = {"person": 1}
with open("labels.pickle", "rb") as f:
    labels = pickle.load(f)
    labels = {v+1:k for k, v in labels.items()}
print(labels)
while True:
    isSuccess, frame = video_cap.read()
    if isSuccess:
        imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(imgGray, 1.1, 3)
        for (x, y, w, h) in faces:
            roi_gray = imgGray[y:y+h, x:x+w]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
            res, confident = recognizer.predict(roi_gray)
            cv2.putText(frame, labels.get(res, "Unknown"), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)            
        cv2.imshow("Face", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break 
video_cap.release()
cv2.destroyAllWindows()