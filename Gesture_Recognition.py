import cv2
import mediapipe as mp
import math
import pyautogui
import time



def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)



# Initialize Mediapipe Hand model
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize PyAutoGUI for controlling media functions
pyautogui.FAILSAFE = False  # Disable fail-safe mode for smooth control

# Initialize the video capture
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Loop until 'q' is pressed
while True:
    success, image = cap.read()
    if not success:
        print("Camera not found!")
        break

    # Flip the image horizontally for a mirrored view
    image = cv2.flip(image, 1)

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Set up the hand tracking
    with mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        # Process the image
        results = hands.process(image_rgb)

        # Check for hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get the coordinates of the thumb and index finger
                thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Convert the coordinates to pixel values
                thumb_x, thumb_y = int(thumb.x * image.shape[1]), int(thumb.y * image.shape[0])
                index_x, index_y = int(index.x * image.shape[1]), int(index.y * image.shape[0])

                # Calculate the distance between thumb and index finger
                distance = calculate_distance(thumb_x, thumb_y, index_x, index_y)

                # Check the distance and perform actions
                if 20 < distance < 30:
                    pyautogui.press('space')  # Pause/Play
                    time.sleep(2)
                elif thumb_x < index_x:
                    pyautogui.press('left')  # Seek backward

                else:
                    pyautogui.press('right')  # Seek forward

                # Draw circles on thumb and index finger
                cv2.circle(image, (thumb_x, thumb_y), 10, (0, 255, 0), -1)
                cv2.circle(image, (index_x, index_y), 10, (0, 0, 255), -1)

    # Display the image
    cv2.imshow("Hand Detection", image)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
