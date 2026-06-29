import cv2
import mediapipe as mp
import pyautogui
import math

pyautogui.FAILSAFE = False

screen_w, screen_h = pyautogui.size()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

prev_x = 0
prev_y = 0
smooth = 7

left_clicked = False
right_clicked = False
dragging = False


def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])


while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame,
                       cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        lm = hand.landmark

        index = (int(lm[8].x*w),
                 int(lm[8].y*h))

        thumb = (int(lm[4].x*w),
                 int(lm[4].y*h))

        middle = (int(lm[12].x*w),
                  int(lm[12].y*h))

        index_mcp = (int(lm[5].x*w),
                     int(lm[5].y*h))

        # cursor position
        cx = screen_w * index[0] / w
        cy = screen_h * index[1] / h

        # smoothing
        cx = prev_x + (cx-prev_x)/smooth
        cy = prev_y + (cy-prev_y)/smooth

        prev_x = cx
        prev_y = cy

        pyautogui.moveTo(cx, cy)

        left_dist = dist(index, thumb)
        right_dist = dist(index, middle)

        # LEFT CLICK
        if left_dist < 35 and not left_clicked:

            pyautogui.click()

            left_clicked = True

        elif left_dist >= 35:

            left_clicked = False

        # RIGHT CLICK
        if right_dist < 25 and not right_clicked:

            pyautogui.rightClick()

            right_clicked = True

        elif right_dist >= 25:

            right_clicked = False

        # DRAG
        if left_dist < 20 and not dragging:

            pyautogui.mouseDown()

            dragging = True

        elif left_dist > 40 and dragging:

            pyautogui.mouseUp()

            dragging = False

        draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        cv2.circle(frame,
                   index,
                   10,
                   (0,255,0),
                   -1)

    cv2.imshow(
        "VizCtrl Mouse",
        frame
    )

    if cv2.waitKey(1) == 27:
        break

if dragging:
    pyautogui.mouseUp()

cap.release()
cv2.destroyAllWindows()