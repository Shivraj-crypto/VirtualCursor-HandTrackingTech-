import cv2
import pyautogui

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam         
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame!")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop through all detected faces
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extract the region of interest (ROI) for eye tracking (within the face region)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Use OpenCV Haar Cascade for detecting eyes within the face region
        eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        eyes = eyes_cascade.detectMultiScale(roi_gray)

        # If two eyes are detected, calculate the average position of both eyes
        if len(eyes) >= 2:
            # Sort eyes by x-coordinate to differentiate left and right eye
            eyes = sorted(eyes, key=lambda e: e[0])

            # Get coordinates of both eyes
            (ex1, ey1, ew1, eh1) = eyes[0]  # Left eye
            (ex2, ey2, ew2, eh2) = eyes[1]  # Right eye

            # Draw rectangles around both eyes
            cv2.rectangle(roi_color, (ex1, ey1), (ex1 + ew1, ey1 + eh1), (0, 255, 0), 2)
            cv2.rectangle(roi_color, (ex2, ey2), (ex2 + ew2, ey2 + eh2), (0, 255, 0), 2)

            # Calculate the center of both eyes
            cx1 = int(ex1 + ew1 / 2)
            cy1 = int(ey1 + eh1 / 2)

            cx2 = int(ex2 + ew2 / 2)
            cy2 = int(ey2 + eh2 / 2)

            # Calculate the average center of both eyes
            avg_cx = (cx1 + cx2) // 2
            avg_cy = (cy1 + cy2) // 2

            # Map the average eye position to screen coordinates
            screen_width, screen_height = pyautogui.size()
            mouse_x = int((avg_cx / w) * screen_width)
            mouse_y = int((avg_cy / h) * screen_height)

            # Move the mouse pointer based on the average eye position
            pyautogui.moveTo(mouse_x, mouse_y)

    # Show the frame with face and eye detection
    cv2.imshow('Eye Tracking - Face Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
