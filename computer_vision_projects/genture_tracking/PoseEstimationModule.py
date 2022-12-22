import cv2
import mediapipe as mp
import time

import resize as rs


class PoseDetector:
    def __init__(self,
                 static_image_mode=False,
                 upper_body_only=False,
                 smooth_landmarks=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.upper_body_only = upper_body_only
        self.smooth_landmarks = smooth_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.static_image_mode,
                                     self.upper_body_only,
                                     self.smooth_landmarks,
                                     self.min_detection_confidence,
                                     self.min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils
        self.results = 0

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def getPositions(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for index, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([index, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

        return lmList


def main():
    cap = cv2.VideoCapture('videos/PoseEstimation/1.mp4')
    detector = PoseDetector()

    cTime = 0
    pTime = 0

    while cap.isOpened():
        success, img = cap.read()
        if success:
            img = rs.resize(img)
            img = detector.findPose(img)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            cv2.putText(img, str(int(fps)), (30, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

            cv2.imshow("Image", img)
            cv2.waitKey(1)
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
