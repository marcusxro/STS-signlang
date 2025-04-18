import pickle
import cv2
import mediapipe as mp
import numpy as np


model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


labels_dict = {
   'A',  'B',  'C',  'D', 'E',
    'F', 'G', 'H', 'I', 'J',
     'K',  'L',  'M',  'N',  'O',
     'P',  'Q',  'R',  'S',  'T',
     'U',  'V',  'W',  'X',  'Y',  'Z'
}

cap = cv2.VideoCapture(0) 

if not cap.isOpened():
    print("Error: Could not open video capture")
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret or frame is None:
        print("Error: Couldn't capture frame")
        break
    
    H, W = frame.shape[:2]
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    

    results = hands.process(frame_rgb)
    
    data_aux = []
    x_ = []
    y_ = []
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style())

            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)
            
            min_x, min_y = min(x_), min(y_)
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min_x)
                data_aux.append(lm.y - min_y)
            
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10
            
            if len(data_aux) == 42:  
                prediction = model.predict([np.asarray(data_aux)])
                predicted_char = prediction[0]

                
    
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                cv2.putText(frame, predicted_char, (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
    

    cv2.imshow('ASL Recognition', frame)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()