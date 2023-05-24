import os
import cv2
import time
import numpy as np
from keras.models import load_model

import warnings
warnings.filterwarnings(action='ignore')

model = load_model('module2.h5')

gestures = {
    0: 'bye',
    1: 'noball',
    2: 'oneshort',
    3: 'out'
}

def predict(gesture):
    img = cv2.resize(gesture, (50, 50))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
    img = img.reshape(1, 50, 50, 3)  # Change 1 to 3 for RGB images
    img = img / 255.0
    prd = model.predict(img)
    index = prd.argmax()
    return gestures[index]

vc = cv2.VideoCapture(0)
rval, frame = vc.read()
old_text = ''
pred_text = ''
count_frames = 0
total_str = ''
flag = False
wicket = 0
score = 0
status = '1'
while True:
    if frame is not None:
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (600, 600))

        cv2.rectangle(frame, (400, 400), (50, 50), (0, 255, 0), 2)

        crop_img = frame[100:300, 100:300]
        grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

        thresh = cv2.threshold(grey, 210, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        blackboard = np.zeros(frame.shape, dtype=np.uint8)
        cv2.putText(blackboard, "Score Board:- ", (30, 40), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 255))
        if count_frames > 20 and pred_text != "":
            total_str += pred_text
            count_frames = 0
        cv2.putText(blackboard, "Wicket " + str(wicket), (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 255))
        cv2.putText(blackboard, "Score " + str(score), (30, 130), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 255))
        f = open('C:/Users/Shantanu J/Desktop/umpire/output.txt', 'r')

        ustatus = f.read()
        f.close()
        print('umpire status', ustatus)
        if flag == True and status == '1' and ustatus == 'umpire':
            pred_text = predict(thresh)
            print(pred_text)

            old_text = pred_text

            if str(pred_text) == 'out':
                res = 'out'
                print(res)
                wicket = wicket + 1
                if wicket == 10:
                     status = '0'
                     flag = False
            elif str(pred_text) == 'noball':
                res = 'noball'
                print(res)
                score = score + 1
            elif str(pred_text) == 'bye':
                res = 'bye'
                print(res)
                score = score + 2
            elif str(pred_text) == 'oneshort':
                res = 'oneshort'
                print(res)
                score = score + 10

            cv2.putText(blackboard, res, (30, 70), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 255))
            cv2.putText(blackboard, "Wicket " + str(wicket), (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 255))
            cv2.putText(blackboard, "Score " + str(score), (30, 130), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 255))
            f = open('result.txt', 'w')
            f.write(str(res))
            f.close()
            if old_text == pred_text:
                count_frames += 1
            else:
                count_frames = 0

        res = np.hstack((frame, blackboard))
        cv2.imshow("image", res)

    rval, frame = vc.read()
    keypress = cv2.waitKey(1)
    flag = False
    if keypress == ord('c'):
        flag = True
    if keypress == ord('q'):
        break

vc.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
vc.release()
