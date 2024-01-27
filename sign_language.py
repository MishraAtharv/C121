import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

finger_tips = [8, 12, 16, 20]
thumb_tip = 4

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, c = img.shape
    results = hands.process(img)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            lm_list = [(lm.x * w, lm.y * h) for lm in hand_landmark.landmark]

            for tip_id in finger_tips:
                if tip_id < len(lm_list):  # Check if landmark index is within range
                    tip_x, tip_y = map(int, lm_list[tip_id])
                    cv2.circle(img, (tip_x, tip_y), 10, (255, 0, 0), cv2.FILLED)

            # Check if enough landmarks are present for finger folding
            if len(lm_list) >= max(finger_tips) + 1:
                finger_fold_status = [lm_list[i][0] < lm_list[i - 2][0] for i in finger_tips[1:]]

                # Check thumb movement for LIKE or DISLIKE
                if all(finger_fold_status):
                    if thumb_tip < len(lm_list) and thumb_tip - 2 < len(lm_list):  # Check if indices are within range
                        if lm_list[thumb_tip][1] < lm_list[thumb_tip - 2][1]:
                            print("LIKE")
                            cv2.putText(img, "LIKE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                                        cv2.LINE_AA)
                        else:
                            print("DISLIKE")
                            cv2.putText(img, "DISLIKE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                        cv2.LINE_AA)

            mp_draw.draw_landmarks(img, hand_landmark,
                                   mp_hands.HAND_CONNECTIONS, mp_draw.DrawingSpec((0, 0, 255), 2, 2),
                                   mp_draw.DrawingSpec((0, 255, 0), 4, 2))

    cv2.imshow("hand tracking", img)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
