import cv2
import os
def takePhoto(name):
 #if new person , create folder 
 base_dir = os.path.dirname(os.path.abspath(__file__))
 img_dir = os.path.join(base_dir, "images")
 persons = os.listdir(img_dir)
 if name not in persons:
     os.mkdir(os.path.join(img_dir, name))
     print(f'folder {name} have been created!')
 #get video capture from webcam
 video_cap = cv2.VideoCapture(0)
 #detect face
 faceCascade = cv2.CascadeClassifier("cascade/haarcascade_frontalface_default.xml")
 count  = 0
 while True:
  isSuccess, frame = video_cap.read()
  if isSuccess:
   imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   faces = faceCascade.detectMultiScale(imgGray, 1.5, 3)
   for(x, y, w, h) in faces:
       count+=1
       image_res = frame[y:y+h, x:x+w]
       cv2.imwrite(f'images/{name}/{name}_{count}.jpg', image_res)
       cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
   cv2.imshow("Scanning", frame)
  if cv2.waitKey(100) & 0xFF == ord("q"):
      break
  elif count >=30:
      break

person = input("Who is he/she: ")
takePhoto(person)