import cv2
import numpy as np


aruco_dicts = {
    "4x4": cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
    "5x5": cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100),
    "6x6": cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
}

# Create detectors for each dictionary
detectors = {key: cv2.aruco.ArucoDetector(aruco_dicts[key], cv2.aruco.DetectorParameters()) 
             for key in aruco_dicts}


# Camera setup

cap = cv2.VideoCapture(0)

# Dummy calibration data
camera_matrix = np.array([[800, 0, 320],
                          [0, 800, 240],
                          [0, 0, 1]], dtype=float)
dist_coeffs = np.zeros((5, 1))


# Main Loop

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    all_corners = []
    all_ids = []
    # Detect markers from all dictionaries
    for key in detectors:
        corners, ids, rejected = detectors[key].detectMarkers(gray)
        if ids is not None:
            all_corners.extend(corners)
            all_ids.extend(ids)

    if all_ids:
        all_ids = np.array(all_ids)
        cv2.aruco.drawDetectedMarkers(frame, all_corners, all_ids)

        # Estimate pose
        marker_length = 0.05  # in meters
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            all_corners, marker_length, camera_matrix, dist_coeffs)

        for i in range(len(all_ids)):
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.03)
            pos = tvecs[i][0]
            cv2.putText(frame, f"ID: {int(all_ids[i])}", (10, 30 + 40*i),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"X:{pos[0]:.2f} Y:{pos[1]:.2f} Z:{pos[2]:.2f}",
                        (10, 60 + 40*i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("ArUco Pose Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
