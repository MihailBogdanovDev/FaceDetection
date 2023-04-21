import cv2
from random import randrange

#Loading pre-trained data from opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


#Getting video from default
webcam = cv2.VideoCapture(0)

while True:

    #Read the current frame
    successfull_frame_read, frame = webcam.read()

    #Making it gray so it works with ml model
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    #Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    #Making it gray so it works with ml model
    #Draw the rectangles
    

    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 5)

    cv2.imshow('Face Detector', frame)
    key = cv2.waitKey(1)
    #Press q or Q to quit
    if key==81 or key==113:
        break
       

webcam.release()
    

"""""
#Detection with image

#Choose image for detection
img = cv2.imread('TwoFacesPhoto.jpg')

#Making it gray so it works with ml model
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

#Draw the rectangles
(x,y,w,h) = face_coordinates[0]

for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256),randrange(256),randrange(256)), 5)



cv2.imshow('Face Detector', img)
cv2.waitKey()
"""""

print("Code Completed")