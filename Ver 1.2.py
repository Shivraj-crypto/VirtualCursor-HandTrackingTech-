import cv2
import numpy as np
import pyautogui
import dlib

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Load face and shape predictors
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('C:/Project/Project Version 1.2/shape_predictor_68_face_landmarks.dat')

# Function to calculate the midpoint between two points
def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

# Get screen size for cursor movement mapping
screen_width, screen_height = pyautogui.size()

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)

        # Get the coordinates for the eyes
        left_eye = (landmarks.part(36).x, landmarks.part(36).y)
        right_eye = (landmarks.part(39).x, landmarks.part(39).y)

        # Calculate the midpoint of the eyes
        eye_center = midpoint(landmarks.part(37), landmarks.part(38))

        # Get the eye region for left eye (using dlib landmarks)
        left_eye_region = frame[landmarks.part(37).y:landmarks.part(41).y, landmarks.part(36).x:landmarks.part(39).x]
        # Convert the region to grayscale
        gray_eye = cv2.cvtColor(left_eye_region, cv2.COLOR_BGR2GRAY)

        # Threshold to detect pupil area
        _, threshold_eye = cv2.threshold(gray_eye, 30, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours (pupil)
        contours, _ = cv2.findContours(threshold_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour which is assumed to be the pupil
            largest_contour = max(contours, key=cv2.contourArea)
            # Get the center of the pupil
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # Draw the pupil center
                cv2.circle(left_eye_region, (cX, cY), 5, (0, 0, 255), -1)

                # Map the pupil position to screen coordinates (invert X-axis for correct mapping)
                pupil_x = np.interp(cX, [0, left_eye_region.shape[1]], [screen_width, 0])  # Reverse mapping for X
                pupil_y = np.interp(cY, [0, left_eye_region.shape[0]], [0, screen_height])

                # Move the cursor based on the pupil's position
                pyautogui.moveTo(pupil_x, pupil_y)

        # Visualize the left eye with the pupil
        cv2.rectangle(frame, (left_eye[0], left_eye[1]), (right_eye[0], right_eye[1]), (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Frame", frame)

    # Exit loop when 'ESC' key is pressed
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
