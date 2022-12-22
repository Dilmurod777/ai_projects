import cv2


def resize(img, height=720):
    h, w, c = img.shape
    ratio = w / h
    img = cv2.resize(img, (int(height * ratio), height))
    return img
