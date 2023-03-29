import numpy as np
import cv2
import time


def main():
    face_cascade = cv2.CascadeClassifier('data\\xml\\haarcascade_frontalface_default.xml')
    mouth_cascade = cv2.CascadeClassifier('data\\xml\\haarcascade_mcs_mouth.xml')
    smile_cascade = cv2.CascadeClassifier('data\\xml\\haarcascade_smile.xml')

    bw_threshold = 127
    weared_mask = "WITH MASK"
    not_weared_mask = "WITHOUT MASK"

    cap = cv2.VideoCapture(0)

    pTime = 0

    while 1:
        ret, img = cap.read()
        img = cv2.flip(img,1)
        wn = "Mask detection"
        cv2.namedWindow(wn, cv2.WINDOW_NORMAL)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (thresh, black_and_white) = cv2.threshold(gray, bw_threshold, 255, cv2.THRESH_BINARY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 10)
        faces_bw = face_cascade.detectMultiScale(black_and_white, 1.1, 10)

        if(len(faces) == 0 and len(faces_bw) == 0):
            cv2.putText(img, "No face found", (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, cv2.LINE_AA)
        else:
            for (x, y, w, h) in faces:
                x1, y1 = x + w, y + h
                couter = 0
                colorLine = (0,0,0)
                color = (0,0,0)       
                mouth = mouth_cascade.detectMultiScale(gray, 2, 10)
                for (mx, my, mw, mh) in mouth:
                    if(x < mx and y < my and y < my < y + h):
                        cv2.rectangle(img, (mx, my), (mx + mw, my + mh), (255, 255, 0), 2)
                        couter+=1
                smile = smile_cascade.detectMultiScale(gray, 2, 15)
                for (sx, sy, sw, sh) in smile:
                    if(x < sx and y < sy):
                        cv2.rectangle(img, (mx, my), (mx + mw, my + mh), (255, 255, 50), 2)
                        couter+=1
                if(couter > 0):
                    cv2.putText(img, not_weared_mask, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
                    colorLine = (59, 59,217)
                    color = (0, 0, 255)
                else:
                    cv2.putText(img, weared_mask, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 1, cv2.LINE_AA)
                    colorLine = (108, 181,60)
                    color = (0, 255, 0)
                    
                cv2.rectangle(img, (x, y), (x1, y1), colorLine, 1)
                cv2.line(img, (x, y), (x + 30, y), color, 3)
                cv2.line(img, (x, y), (x, y+30), color, 3)
                cv2.line(img, (x1, y), (x1 - 30, y), color, 3)
                cv2.line(img, (x1, y), (x1, y+30), color, 3)
                cv2.line(img, (x, y1), (x + 30, y1), color, 3)
                cv2.line(img, (x, y1), (x, y1 - 30), color, 3)
                cv2.line(img, (x1, y1), (x1 - 30, y1), color, 3)
                cv2.line(img, (x1, y1), (x1, y1 - 30), color, 3)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        
        #cv2.putText(img, f'FPS: {int(fps)}', (10, 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        cv2.imshow(wn, img)
        cv2.imshow("ddwa", black_and_white)
        
        k = cv2.waitKey(1)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()