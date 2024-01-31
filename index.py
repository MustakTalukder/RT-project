import cv2
from deepface import DeepFace

face_cascade_data = cv2.CascadeClassifier('/Users/mustak/Desktop/uni/RT/project/haarcascade_frontalface_default.xml')

def captureVideo():
    return cv2.VideoCapture(0)

def faceDetaction(frame):
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    return face_cascade_data.detectMultiScale(gray,1.1,4)

def emotionDetaction(frame):
    print(frame)
    result = DeepFace.analyze(img_path = frame , actions=['emotion'], enforce_detection=False )
    emotion = result[0]['dominant_emotion']

    return str(emotion)

def showResultInlive(frame,txt):
    cv2.putText(frame,txt,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
    cv2.imshow('frame',frame)



captureVideoFrame = captureVideo()

while True:
    ret,frame = captureVideoFrame.read()

    allFaces = faceDetaction(frame)
    for (x,y,w,h) in allFaces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)

    emotionName = emotionDetaction(frame)
    # txt = "Face"

    showResultInlive(frame,emotionName)

    # cnt+q for quit
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

captureVideoFrame.release()
cv2.destroyAllWindows('q')