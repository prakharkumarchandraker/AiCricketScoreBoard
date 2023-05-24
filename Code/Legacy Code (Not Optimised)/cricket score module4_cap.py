#............................LiveCamera with gesture recognize the Indian sign.........................#

import os
import cv2
import time
import numpy as np
from keras.models import load_model

import warnings
warnings.filterwarnings(action = 'ignore')


model = load_model('CNN_model.h5')

gestures = {
    1:'Out',
    2:'No_ball',
    3:'Six',
    4:'Two',
    5:'None'
}

def predict(gesture):
    img = cv2.resize(gesture, (50,50))
    img = img.reshape(1,50,50,1)
    img = img/255.0
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
wicket=0
score=0
status='1'
while True:
    #time.sleep(10)
   
    if frame is not None: 
        
        frame = cv2.flip(frame, 1)
        frame = cv2.resize( frame, (600,600) )
        
        cv2.rectangle(frame, (400,400), (50,50), (0,255,0), 2)
        
        crop_img = frame[100:300, 100:300]
        grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        
        thresh = cv2.threshold(grey,210,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
      
        blackboard = np.zeros(frame.shape, dtype=np.uint8)
        cv2.putText(blackboard, "Score Board:- ", (30, 40), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 255))
        if count_frames > 20 and pred_text != "":
            total_str += pred_text
            count_frames = 0
        cv2.putText(blackboard, "Wicket "+str(wicket), (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 255))
        cv2.putText(blackboard, "Score "+str(score), (30, 130), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 255))
        f = open('C:/Users/admin/Desktop/umpire/output.txt','r')
        ustatus=f.read()
        f.close()
        print('umpire status',ustatus)
        if flag == True and status=='1' and ustatus=='umpire':
            
            pred_text = predict(thresh)
            print(pred_text)
            
            old_text = pred_text
            #f = open('H:/code check/Cricket_Score/output.txt','w')
            #f.write(str(pred_text))
            #f.close()

            if str(pred_text)=='Out':
                res = 'Out'
                print(res)
                wicket=wicket+1
                if wicket==10:
                    status='0'
                    flag=False
            elif str(pred_text)=='No_ball':
                res = 'No_ball'
                print(res)
                score=score+1
            elif str(pred_text)=='None':
                res = 'None'
                print(res)
            elif str(pred_text)=='Six':
                res = 'Six'
                print(res)
                score=score+6
            elif str(pred_text)=='Two':
                res = 'Two'
                print(res)
                score=score+2
            '''
            
            elif str(pred_text)=='Four':
                res = 'Four'
                print(res)
                score=score+4
            '''
            
            print('res',res)
            
            cv2.putText(blackboard, res, (30, 70), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 255))
            cv2.putText(blackboard, "Wicket "+str(wicket), (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 255))
            cv2.putText(blackboard, "Score "+str(score), (30, 130), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 255))
            f = open('result.txt','w')
            f.write(str(res))
            f.close()
            if old_text == pred_text:
                count_frames += 1                
            else:
                count_frames = 0
        #print('*************',pred_text)
            #f = open('H:/code check/Cricket_Score/result.txt','w')
            #f.write(str(res))
            #f.close()

            #cv2.putText(blackboard, total_str, (30, 80), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0))
        res = np.hstack((frame, blackboard))
        
        cv2.imshow("image", res)
        #cv2.imshow("hand", thresh)
        #time.sleep(1)
    rval, frame = vc.read()
    keypress = cv2.waitKey(1)
    flag=False
    if keypress == ord('c'):
        flag = True
    if keypress == ord('q'):
        break

vc.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
vc.release()

