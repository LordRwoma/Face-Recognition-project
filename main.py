import cv2

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        id,pred = clf.predict(gray_img[y:y+h, x:x+w])
        confidence = int(100*(1-pred/300))

        if confidence>72:
            if id==1:
                cv2.putText(img, "Safira", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            if id==2:
                cv2.putText(img, "Wildan", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            if id==3:
                cv2.putText(img, "non", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "ANOMALI", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)
        coords = [x, y, w, h]

    return coords

def recognize(img, clf, faceCascade):
     coords = draw_boundary(img, faceCascade, 1.1, 10, (255,255,255), "Face", clf)
     return img


xml_path = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(xml_path)
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.xml")

capture = cv2.VideoCapture(0)
img_id = 0
while True:
    ret, img = capture.read()
    img = recognize(img, clf, faceCascade)
    cv2.imshow("face detection", img)
    img_id += 1

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
capture.release()
cv2.destroyAllWindows()