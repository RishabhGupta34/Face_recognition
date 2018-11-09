import cv2
cap=cv2.VideoCapture(0)
Face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
while True:
    ret,frame=cap.read()
    if not ret:
        continue
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
# cv2.imshow("Video frame",frame)
# cv2.imshow("Gray frame",gray_frame)
    faces=Face_cascade.detectMultiScale(gray_frame,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow("Vf",frame)
    key_pressed=cv2.waitKey(1) & 0xFF
    if key_pressed==ord('q'):
        break
cap.release()