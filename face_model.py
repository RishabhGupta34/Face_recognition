import os,cv2
import numpy as np

def distance(v1,v2):
	return np.sqrt(((v1-v2)**2).sum())

def knn(train,test,k=5):
	distances=[]
	for t in range(train.shape[0]):
		ix=train[t,:-1]
		iy=train[t,-1]
		d=distance(ix,test)
		distances.append([d,iy])
	distances=sorted(distances,key=lambda x:x[0])[:k]
	labels=np.array(distances)[:,-1]
	count=np.unique(labels,return_counts=True)
	index=np.argmax(count[1])
	return count[0][index]

face_data=[]
labels=[]
data_path='./data/'
label_to_name={}
class_id=0
files=os.listdir(data_path)
for i in range(len(files)):
	if(".npy" in files[i]):
		label_to_name[class_id]=files[i][:-4]
		data=np.load(data_path+files[i])
		face_data.append(data)
		target=class_id*np.ones((data.shape[0],))
		class_id+=1
		labels.append(target)
face_data=np.concatenate(face_data,axis=0)
labels=np.concatenate(labels,axis=0).reshape((-1,1))

train_data=np.concatenate((face_data,labels),axis=1)


cap=cv2.VideoCapture(0)
Face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
while True:
    ret,frame=cap.read()
    if not ret:
        continue
    faces=Face_cascade.detectMultiScale(frame,1.3,5)
    if len(faces)==0:
    	continue
    for face in faces:
    	offset=10
    	x,y,w,h=face
    	test=frame[y-offset:y+h+offset,x-offset:x+w+offset]
    	test=cv2.resize(test,(100,100))
    	label=knn(train_data,test.flatten())
    	pred_name=label_to_name[label]
    	cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
    	cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
    cv2.imshow("Frame",frame)
    key_pressed=cv2.waitKey(1) & 0xFF
    if key_pressed==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()