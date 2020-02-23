import cv2
car_cascade = cv2.CascadeClassifier('cars.xml')

video = cv2.VideoCapture('car12.mp4')

# print(type(video))
while True:
     check,frame=video.read()
     gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
     cars=car_cascade.detectMultiScale(gray,scaleFactor=1.05,minNeighbors=25)

     for x,y,w,h in cars:
         rectangle= cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
         
     cv2.imshow('Gray Image',frame)
     cv2.waitKey(1)
cv2.destroyAllWindows()
video.release()
