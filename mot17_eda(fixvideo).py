import pandas as pd 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

col_list = ["frame_number", "identity_number","left","top","width","height","score","class","visibility"]
#col list collumn yani bütün gt textindeki metinin içini alıyor ve bir değişkene atıyor

data = pd.read_csv("gt.txt", names = col_list)

pedestrian = data[data["class"] == 1]
car = data[data["class"] == 3]

video_path = "MOT17-13-SDP.mp4"

cap = cv2.VideoCapture(video_path)

id1 = 35
id2 = 133
id3 = 29

numberOfImage = np.max(data["frame_number"])
fps = 25
bound_box_list = []
#kutucuklar depolanıyor

for i in range(numberOfImage-1):
    ret, frame = cap.read()
    
    if ret:
        
        frame = cv2.resize(frame, dsize=(960,540))
        #boyutunu düşürüyoruz vidin (frame anlık videonun framei)
        
        filter_id2 = np.logical_and(pedestrian["frame_number"] == i+1, pedestrian["identity_number"]==id2)
        
        if len(pedestrian[filter_id2]) != 0:
            
            x = int(pedestrian[filter_id2].left.values[0]/2)
            y = int(pedestrian[filter_id2].top.values[0]/2)
            w = int(pedestrian[filter_id2].width.values[0]/2)
            h = int(pedestrian[filter_id2].height.values[0]/2)
            
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
            cv2.circle(frame, (int(x+w/2),int(y+h/2)), 2, (0,0,255), -1)
            
            #frame x,y,w,h ve center_x, center_y basit matematik
            bound_box_list.append([i, x,y,w,h,int(x+w/2),int(y+h/2)])
            
        cv2.putText(frame, "Frame num:"+str(i+1), (10,30), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        cv2.imshow("wido", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"): break
    else: break
cap.release()
cv2.destroyAllWindows()
            
