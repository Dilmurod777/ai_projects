import cv2
import time
import os
import resize as rs
import HandTrackingModule as htm

wCam, hCam = 1200, 720
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

cTime = 0
pTime = 0

folderPath = 'images/FingerCounter'
folderContent = os.listdir(folderPath)
overlayImages = []
for item in folderContent:
    image = cv2.imread(f'{folderPath}/{item}')
    overlayImages.append(image)

detector = htm.HandDetector(min_detection_confidence=0.75)
tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()

    if success:
        img = rs.resize(img)

        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)

        if len(lmList) != 0:
            fingers = []

            if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            for id in range(1, len(tipIds)):
                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            totalFingers = fingers.count(1)

            h, w, c = overlayImages[totalFingers].shape
            img[0:h, 0:w] = overlayImages[0]

            cv2.rectangle(img, (20, 255), (170, 425), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        h, w, c = overlayImages[0].shape
        cv2.putText(img, str(int(fps)), (w + 30, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
    else:
        break

cap.release()
cv2.destroyAllWindows()
