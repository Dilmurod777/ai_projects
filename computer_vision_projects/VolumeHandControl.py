import cv2
import time
import math
import numpy as np
import resize as rs
import HandTrackingModule as htm

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wCam, hCam = 1200, 720

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

cTime = 0
pTime = 0

detector = htm.HandDetector(min_detection_confidence=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volumeRange = volume.GetVolumeRange()
minVolume = volumeRange[0]
maxVolume = volumeRange[1]

minLength = 50
maxLength = 250

while True:
    success, img = cap.read()
    img = rs.resize(img)

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        length = math.hypot(x2 - x1, y2 - y1)
        if length < 50:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

        newVolume = int(np.interp(length, [minLength, maxLength], [minVolume, maxVolume]))
        newVolumeBar = int(np.interp(length, [minLength, maxLength], [400, 150]))
        newVolumeBarPercentage = int(np.interp(length, [minLength, maxLength], [0, 100]))
        volume.setMasterVolumeLevel(newVolume, None)
    else:
        newVolumeBar = 400
        newVolumeBarPercentage = 0

    cv2.rectangle(img, (40, 150), (75, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (40, newVolumeBar), (75, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{newVolumeBarPercentage}%', (40, 450), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    if success:
        cv2.putText(img, str(int(fps)), (30, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)
    else:
        break

cap.release()
cv2.destroyAllWindows()
