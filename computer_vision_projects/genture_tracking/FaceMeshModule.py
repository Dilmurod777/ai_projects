import cv2
import mediapipe as mp
import time

import resize as rs


class FaceMeshDetector:
    def __init__(self, static_image_mode=False, max_num_faces=4,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.static_image_mode,
                                                 self.max_num_faces,
                                                 self.min_detection_confidence,
                                                 self.min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=2, circle_radius=1)
        self.results = 0

    def findFaceMesh(self, img, draw=True):
        img = rs.resize(img)
        h, w, c = img.shape
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)

        faces = []
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACE_CONNECTIONS, self.drawSpec)

                face = []
                for lmIndex, lm in enumerate(faceLms.landmark):
                    x, y = int(lm.x * w), int(lm.y * h)
                    face.append([lmIndex, x, y])
                faces.append(face)

        return img, faces


def main():
    cap = cv2.VideoCapture('videos/FaceMesh/5.mp4')
    detector = FaceMeshDetector()
    cTime = 0
    pTime = 0

    while cap.isOpened():
        success, img = cap.read()
        if success:
            img, faces = detector.findFaceMesh(img)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
            cv2.imshow('Image', img)
            cv2.waitKey(1)
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
