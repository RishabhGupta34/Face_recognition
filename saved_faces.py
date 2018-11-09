import numpy as np
import cv2,os
cam=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
face_data=[]
path="./data/"
if not os.path.exists(path):
	os.mkdir(path)
file_name=input("Enter the name")
cnt=0
while True:
	ret,frame=cam.read()
	if ret==False:
		break
	faces=face_cascade.detectMultiScale(frame,1.3,5)
	if len(faces)==0:
		continue
	faces=sorted(faces,key=lambda x:x[2]*x[3])
	x,y,w,h=faces[-1]
	cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
	offset=10
	face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
	face_section=cv2.resize(face_section,(100,100))
	face_data.append(face_section)
	cv2.imshow("face_section",face_section)
	cnt+=1
	cv2.imshow("face",frame)
	print(cnt)
	key=cv2.waitKey(1) & 0xFF
	if key==ord('q'):
		break	
face_data=np.asarray(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)
np.save(path+file_name,face_data)
cam.release()
cv2.destroyAllWindows()