import cv2

cap = cv2.VideoCapture(0)

cascade = cv2.CascadeClassifier('face.xml')#adaboost
#method for combining more complex classifiers such that it discards the negative inputs which increase the accuracy and reduce the
#computation power and all the computation power works on positive inputs.MORE EFFICIENT

while cap.isOpened():
    result , frame = cap.read()
    if result:
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces =cascade.detectMultiScale(gray ,1.3 , 5 , 0 , minSize=(120,120) , maxSize=(350,350))
        for (x,y,w,h) in faces:
            print(x,y,w,h)
            cv2.rectangle(frame , (x,y) , (x+w , y+h) , (0,255,0)  , 5)
        cv2.imshow("frame" , frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
#1st --> haar feature selection ---- it produce a single value by taking
# the difference of sum of the intensities of dark region and sum of the intensities of light region.
#2nd --> Converting image to gray scale.
