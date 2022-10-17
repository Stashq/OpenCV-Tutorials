import cv2
import numpy as np

vid = cv2.VideoCapture(0)

while True:
    ret, frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
    corners = np.int0(corners)

    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) == ord("q"):
        break

vid.release()
cv2.destroyAllWindows()
