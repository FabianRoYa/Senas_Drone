import cv2
import time
import numpy as np
from djitellopy import Tello
import mediapipe as mp

def callback(x):
    pass
def get_x(landmarks):
    x = []
    x.append(landmarks[mp_hands.HandLandmark.WRIST].x)
    x.append(landmarks[mp_hands.HandLandmark.THUMB_CMC].x)
    x.append(landmarks[mp_hands.HandLandmark.THUMB_MCP].x)
    x.append(landmarks[mp_hands.HandLandmark.THUMB_IP].x)
    x.append(landmarks[mp_hands.HandLandmark.THUMB_TIP].x)
    x.append(landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].x)
    x.append(landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].x)
    x.append(landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP].x)
    x.append(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x)
    x.append(landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x)
    x.append(landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x)
    x.append(landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x)
    x.append(landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x)
    x.append(landmarks[mp_hands.HandLandmark.RING_FINGER_MCP].x)
    x.append(landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].x)
    x.append(landmarks[mp_hands.HandLandmark.RING_FINGER_DIP].x)
    x.append(landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].x)
    x.append(landmarks[mp_hands.HandLandmark.PINKY_MCP].x)
    x.append(landmarks[mp_hands.HandLandmark.PINKY_PIP].x)
    x.append(landmarks[mp_hands.HandLandmark.PINKY_DIP].x)
    x.append(landmarks[mp_hands.HandLandmark.PINKY_TIP].x)
    return x

def get_y(landmarks):
    y = []
    y.append(landmarks[mp_hands.HandLandmark.WRIST].y)
    y.append(landmarks[mp_hands.HandLandmark.THUMB_CMC].y)
    y.append(landmarks[mp_hands.HandLandmark.THUMB_MCP].y)
    y.append(landmarks[mp_hands.HandLandmark.THUMB_IP].y)
    y.append(landmarks[mp_hands.HandLandmark.THUMB_TIP].y)
    y.append(landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].y)
    y.append(landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y)
    y.append(landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP].y)
    y.append(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y)
    y.append(landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y)
    y.append(landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y)
    y.append(landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y)
    y.append(landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y)
    y.append(landmarks[mp_hands.HandLandmark.RING_FINGER_MCP].y)
    y.append(landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y)
    y.append(landmarks[mp_hands.HandLandmark.RING_FINGER_DIP].y)
    y.append(landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y)
    y.append(landmarks[mp_hands.HandLandmark.PINKY_MCP].y)
    y.append(landmarks[mp_hands.HandLandmark.PINKY_PIP].y)
    y.append(landmarks[mp_hands.HandLandmark.PINKY_DIP].y)
    y.append(landmarks[mp_hands.HandLandmark.PINKY_TIP].y)
    return y

def open_right(x, y):
    return y[4] < y[3] and x[4] < x[5] and x[20] > x[16] and y[8] < y[5] and y[12] < y[9] and y[16] < y[13] and y[20] < y[17]

def open_left(x, y):
    return y[4] < y[3] and x[4] > x[5] and x[20] < x[16] and y[8] < y[5] and y[12] < y[9] and y[16] < y[13] and y[20] < y[17]

def go_r(x, y):
    return x[4] > x[3] and x[8] > x[7] and x[12] > x[11] and x[16] > x[15] and x[20] > x[19] and y[20] < y[19]

def go_l(x, y):
    return x[4] < x[3] and x[8] < x[7] and x[12] < x[11] and x[16] < x[15] and x[20] < x[19] and y[20] < y[19] 

def index_middle_up(x, y):     
    return y[8] < y[7] and y[12] < y[11] and y[16] > y[13] and y[20] > y[17]

def index_up(x, y):
    return y[8] < y[7] and y[12] > y[9] and y[16] > y[13] and y[20] > y[17]

def closed(x, y):
    return y[4] > y[10] and y[4] < y[11] and x[4] > x[7] and x[4] < x[14] and y[8] > y[5] and y[12] > y[9] and y[16] > y[13] and y[20] > y[17]

def thumb_up(x, y):
    return y[4] < y[6] and y[8] > y[5] and y[12] > y[9] and y[16] > y[13] and y[20] > y[17]

def thumb_down(x, y):
    return y[4] > y[3] and y[8] > y[5] and y[12] > y[9] and y[16] > y[13] and y[20] > y[17]

def thumb_pinky_up(x, y):
    return y[4] < y[3] and y[20] < y[19] and y[8] > y[5] and y[12] > y[9] and y[16] > y[13]

def four_fingers_up(x, y):
    return y[8] < y[7] and y[12] < y[11] and y[16] < y[15] and y[20] < y[19] and x[4] < x[9] and x[4] > x[17] and y[20] > y[15]

def horizontal(x, y):
    return y[8] < y[9] and y[8] < y[10] and y[8] < y[11] and y[8] < y[12] and y[12] < y[13] and y[12] < y[14] and y[12] < y[15] and y[12] < y[16] and y[16] < y[17] and y[16] < y[18] and y[16] < y[19] and y[16] < y[20]

def is_thumb_up(x, y):
    return (y[4] < y[3] and not go_left(x, y) and not go_right(x, y))

def go_left(x, y):
    return (x[8] < x[7] and x[7] < x[6] and x[6] < x[5] and horizontal (x, y))

def go_right(x, y):
    return (x[8] > x[7] and x[7] > x[6] and x[6] > x[5] and horizontal (x, y))

frame_source = 1
if frame_source == 0:
    capture = cv2.VideoCapture(0)
elif frame_source == 1:
    capture = cv2.VideoCapture(0)
    drone = Tello()
    drone.connect()
    drone.left_right_velocity = 0  
    drone.forward_backward_velocity = 0
    drone.up_down_velocity = 0
    drone.yaw_velocity = 0
    drone.status = 0
    drone.height = 0
    drone.speed = 50
    drone.height_lim = 50
h, w = 500, 500
in_speed = 50
in_height_lim = 50
cv2.namedWindow('Trackbars')
cv2.resizeWindow('Trackbars', (500, 100))
cv2.createTrackbar('Speed', 'Trackbars', 0, 100, callback)
cv2.createTrackbar('Height limit', 'Trackbars', 50, 300, callback)
cv2.setTrackbarPos('Speed', 'Trackbars', in_speed)
cv2.setTrackbarPos('Height limit', 'Trackbars', in_height_lim)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def main():
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        print("Main program running now")
        manual = True
        sign = ""
        while True:
            ret, img = capture.read()
            if not ret:
                print("No se pudo obtener el frame de la webcam.")
                continue
            img = cv2.flip(img, 1)
            img = cv2.resize(img, (500, 500))
            img_tracking = img.copy()
            
            results = hands.process(img)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(img_tracking, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    x = get_x(hand_landmarks.landmark)
                    y = get_y(hand_landmarks.landmark)
                    if go_left(x, y):
                        sign = "left"
                    elif go_right(x, y):
                        sign = "right"
                    elif thumb_up(x, y):
                        sign = "up"
                    elif thumb_down(x, y):
                        sign = "down"
                    elif open_right(x, y):
                        sign = "open right"
                    elif open_left(x, y):
                        sign = "open left"
                    elif index_middle_up(x, y):
                        sign = "2"
                    elif index_up(x, y):
                        sign = "1"
                    elif closed(x, y):
                        sign = "closed"
                    elif thumb_pinky_up(x, y):
                        sign = "pinky"
                    elif four_fingers_up(x, y):
                        sign = "4"
                    else:
                        sign = ""
                    cv2.putText(img_tracking, ("Sign: " + sign), (0, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    
            key = cv2.waitKey(1) & 0xFF
            if frame_source == 1:
                drone.speed = cv2.getTrackbarPos('Speed', 'Trackbars')
                drone.height_lim = cv2.getTrackbarPos('Height limit', 'Trackbars')
                cv2.putText(img_tracking, 'Battery:  ' + str(drone.get_battery()), (0, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 3)
                if drone.get_battery() < 25:
                    cv2.putText(img_tracking, 'Low level', (0, 70), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
                if drone.get_battery() <= 15:
                    cv2.putText(img_tracking, 'Critical level', (0, 90), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
                if key == 109:          # Presionar m para modo manual
                    manual = True
                    print("Modo Manual ACtivado")
                elif key == 110:        # Presionar n para modo automático
                    manual = False
                    print("Modo Atomatico ACtivado (Senas)")

                if manual == True:
                    if key == 116:      # Presionar r para despegar
                        if drone.get_battery() >= 5:
                            if drone.status == 0:
                                drone.status = 1
                                drone.takeoff()
                                print("Despegando")
                    elif key == 108:    # Presionar l para aterrizar
                        if drone.status == 1:
                            drone.status = 0
                            drone.land()
                            print("Aterrizando")
                    elif key == 104:    # Presionar h para mantener su posición.
                        drone.left_right_velocity = 0 
                        drone.forward_backward_velocity = 0
                        drone.up_down_velocity = 0
                        drone.yaw_velocity = 0
                        print("Control Activado")
                    elif key == 119:    # Presionar w para moverse hacia delante
                        drone.left_right_velocity = 0 
                        drone.forward_backward_velocity = drone.speed
                        drone.up_down_velocity = 0
                        drone.yaw_velocity = 0 
                    elif key == 115:    # Presionar s para moverse hacia atrás. 
                        drone.left_right_velocity = 0 
                        drone.forward_backward_velocity = -drone.speed
                        drone.up_down_velocity = 0
                        drone.yaw_velocity = 0
                    elif key == 97:     # Presionar a para moverse a la izquierda
                        drone.left_right_velocity = -drone.speed
                        drone.forward_backward_velocity = 0
                        drone.up_down_velocity = 0
                        drone.yaw_velocity = 0
                    elif key == 100:    # Presionar d para moverse a la derecha
                        drone.left_right_velocity = drone.speed
                        drone.forward_backward_velocity = 0
                        drone.up_down_velocity = 0
                        drone.yaw_velocity = 0
                    elif key == 101:    # Presionar e para subir.
                        drone.left_right_velocity = 0
                        drone.forward_backward_velocity = 0
                        drone.up_down_velocity = drone.speed
                        drone.yaw_velocity = 0
                    elif key == 114:    # Presionar r para bajar.
                        drone.left_right_velocity = 0
                        drone.forward_backward_velocity = 0
                        drone.up_down_velocity = -drone.speed
                        drone.yaw_velocity = 0
                    elif key == 122:    # Presionar z para girar a la izquierda
                        drone.left_right_velocity = 0
                        drone.forward_backward_velocity = 0
                        drone.up_down_velocity = 0
                        drone.yaw_velocity = -drone.speed
                    elif key == 120:    # Presionar x para girar a la derecha
                        drone.left_right_velocity = 0
                        drone.forward_backward_velocity = 0
                        drone.up_down_velocity = 0
                        drone.yaw_velocity = drone.speed
                    else:
                        drone.left_right_velocity = 0
                        drone.forward_backward_velocity = 0
                        drone.up_down_velocity = 0
                        drone.yaw_velocity = 0

                else:
                    if sign == "open right":
                        if drone.status == 1:
                            drone.status = 0
                            drone.land()
                    elif sign == "open left":
                        if drone.status == 0:
                            drone.status = 1
                            drone.takeoff()
                    elif sign == "right":
                        drone.up_down_velocity = 0
                        drone.left_right_velocity = -drone.speed
                        drone.forward_backward_velocity = 0
                        drone.yaw_velocity = 0
                    elif sign == "left":
                        drone.up_down_velocity = 0
                        drone.left_right_velocity = drone.speed
                        drone.forward_backward_velocity = 0
                        drone.yaw_velocity = 0
                    elif sign == "2":
                        drone.up_down_velocity = 0
                        drone.left_right_velocity = 0
                        drone.forward_backward_velocity = 0
                        drone.yaw_velocity = -drone.speed
                    elif sign == "1":
                        drone.up_down_velocity = 0
                        drone.left_right_velocity = 0
                        drone.forward_backward_velocity = 0
                        drone.yaw_velocity = drone.speed
                    elif sign == "closed":
                        drone.up_down_velocity = 0
                        drone.left_right_velocity = 0
                        drone.forward_backward_velocity = 0
                        drone.yaw_velocity = 0
                    elif sign == "up":
                        drone.up_down_velocity = drone.speed
                        drone.left_right_velocity = 0
                        drone.forward_backward_velocity = 0
                        drone.yaw_velocity = 0
                    elif sign == "down":
                        drone.up_down_velocity = -drone.speed
                        drone.left_right_velocity = 0
                        drone.forward_backward_velocity = 0
                        drone.yaw_velocity = 0
                    elif sign == "pinky":
                        drone.up_down_velocity = 0
                        drone.left_right_velocity = 0
                        drone.forward_backward_velocity = drone.speed
                        drone.yaw_velocity = 0
                    elif sign == "4":
                        drone.up_down_velocity = 0
                        drone.left_right_velocity = 0
                        drone.forward_backward_velocity = -drone.speed
                        drone.yaw_velocity = 0
                    else:
                        drone.up_down_velocity = 0
                        drone.left_right_velocity = 0
                        drone.forward_backward_velocity = 0
                        drone.yaw_velocity = 0

                if (drone.get_height() > drone.height_lim):
                    drone.up_down_velocity = -drone.speed
                    drone.left_right_velocity = 0
                    drone.forward_backward_velocity = 0
                    drone.yaw_velocity = 0
                drone.send_rc_control(drone.left_right_velocity, drone.forward_backward_velocity, drone.up_down_velocity, drone.yaw_velocity)           
            cv2.imshow("Image", img_tracking)
            if key == 113:              # Presionar q para finalizar
                cv2.destroyAllWindows()
                if frame_source == 1:
                    drone.end()
                break

try:
    main()
except KeyboardInterrupt:
    print('KeyboardInterrupt exception is caught')
    cv2.destroyAllWindows()
    if frame_source == 1:
        drone.land()
        drone.end()
else:
    print('No exceptions are caught')