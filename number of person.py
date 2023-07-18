import cv2
import numpy as np
cap = cv2.VideoCapture(0)
cascade = cv2.CascadeClassifier('face.xml')#adaboost
#method for combining more complex classifiers such that it discards the negative inputs which increase the accuracy and reduce the
#computation power and all the computation power works on positive inputs.MORE EFFICIENT
while True:#cap.isOpened():
    result , frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray ,1.3 , 5 , 0)
    num = 0
    for (x,y,w,h) in faces:
        #x,y = face.left(),face.top()
        #hi,wi = face.right(),face.bottom()
        print(x,y,w,h)
        cv2.rectangle(frame,(x,y),(h,w),(0,0,255),2)
        num = num+1
        cv2.putText(frame,'face'+str(num),(x-12,y-12), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    cv2.imshow('faces',frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
'''
        for (x,y,w,h) in faces:
            print(x,y,w,h)
            cv2.rectangle(frame , (x,y) , (x+w , y+h) , (0,255,0)  , 5)
        cv2.imshow("frame" , frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
'''