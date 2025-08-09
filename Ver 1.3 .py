import cv2
import numpy as np
import pyautogui
import mediapipe as mp

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Get screen size for cursor movement mapping
screen_width, screen_height = pyautogui.size()

# Function to check if the index finger and thumb are touching
def is_thumb_index_touching(hand_landmarks):
    # Get the landmarks for the index finger tip and the thumb tip
    index_tip = hand_landmarks.landmark[8]
    thumb_tip = hand_landmarks.landmark[4]
    
    # Calculate the distance between the index finger tip and thumb tip
    distance = np.linalg.norm(np.array([index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y]))
    
    return distance < 0.05  # Adjust threshold as needed

# Function to map the index finger tip position to screen coordinates
def map_to_screen(index_tip, screen_width, screen_height):
    # Convert the index tip's normalized coordinates to screen coordinates
    x = int(index_tip.x * screen_width)
    y = int(index_tip.y * screen_height)
    
    # Ensure the cursor stays within screen bounds
    x = max(0, min(screen_width - 1, x))
    y = max(0, min(screen_height - 1, y))
    
    return x, y

# Variables for cursor movement threshold (avoid shaking)
last_x, last_y = None, None
movement_threshold = 5  # Adjust this value for smoother movement

# Smoothing variables
smooth_factor = 0.5  # Smoothing factor for cursor movement

# Variables to track previous positions for scrolling
last_index_y = None
scroll_threshold = 0.05  # Distance threshold for scrolling action

while True:
    # Capture frame-by-frame
    _, frame = cap.read()
    
    # Flip the frame horizontally for better user experience
    frame = cv2.flip(frame, 1)
    
    # Convert the BGR frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and get hand landmarks
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the position of the index finger tip
            index_finger_tip = hand_landmarks.landmark[8]
            # Get the position of the middle finger tip
            middle_finger_tip = hand_landmarks.landmark[12]
            
            # Map index finger tip to screen coordinates
            x, y = map_to_screen(index_finger_tip, screen_width, screen_height)
            
            # Apply smoothing to the cursor movement
            if last_x is not None and last_y is not None:
                x = int(last_x + (x - last_x) * smooth_factor)
                y = int(last_y + (y - last_y) * smooth_factor)

            # Move the cursor only if the movement exceeds a threshold
            if last_x is None or abs(last_x - x) > movement_threshold or abs(last_y - y) > movement_threshold:
                pyautogui.moveTo(x, y)
                last_x, last_y = x, y
            
            # Check if the index finger and thumb are touching
            if is_thumb_index_touching(hand_landmarks):
                pyautogui.click()  # Perform left click
            
            # Track scrolling based on index and middle finger movement
            if last_index_y is not None:
                index_finger_y = index_finger_tip.y
                # If the vertical movement between index and middle fingers exceeds threshold
                if abs(index_finger_y - last_index_y) > scroll_threshold:
                    if index_finger_y < last_index_y:
                        pyautogui.scroll(-50)  # Scroll down
                    else:
                        pyautogui.scroll(50)   # Scroll up
            last_index_y = index_finger_tip.y  # Update last index finger position
            
            # Draw landmarks and connections (optional, just for visualization)
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Draw circles around the thumb and index tips (for debugging)
            thumb_tip = hand_landmarks.landmark[4]
            cv2.circle(frame, (int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])), 10, (0, 255, 0), 3)
            cv2.circle(frame, (int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0])), 10, (0, 0, 255), 3)

    # Show the resulting frame
    cv2.imshow("Hand Cursor Control", frame)

    # Exit the loop when 'ESC' key is pressed
    key = cv2.waitKey(1)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()
