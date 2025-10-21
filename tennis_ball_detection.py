import cv2
import numpy as np


cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Blur the frame to reduce noise
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)

    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])


    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)


    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)


    res = cv2.bitwise_and(frame, frame, mask=mask)


    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)


    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=50, param2=30, minRadius=10, maxRadius=150)


    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]


            cv2.circle(frame, center, radius, (0, 255, 0), 3)

            cv2.circle(frame, center, 5, (0, 0, 255), -1)

            cv2.putText(frame, f"({i[0]}, {i[1]})", (i[0] - 40, i[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


    cv2.imshow('Tennis Ball Detection', frame)
    cv2.imshow('Mask', mask)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
