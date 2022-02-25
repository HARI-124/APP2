import datetime
import cv2,time
from datetime import datetime
from cv2 import VideoCapture
import pandas 
first_frame= None
status_list=[None,None]
times=[]
df = pandas.DataFrame(columns=["start","end"])
cap = cv2.VideoCapture(0)


while True:
    result,frame=cap.read()
    stat=0
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21),0)
    
    
    if first_frame is None:
        first_frame = gray# only one value will be stored 
        #in the first iteration of loop first_frame will be None and after that firast frame is
        #first frame is stored in that.
        continue # after the continue the loop will start again
    
    differ_image =cv2.absdiff(first_frame,gray)
    thresh_delta=cv2.threshold(differ_image,30,255,cv2.THRESH_BINARY)[1]
    thresh_delta=cv2.dilate(thresh_delta,None, iterations=3)# here we use dilation
    
    #now we will use contours to detect the outlines
    # contours are basically two types drawand find
    (cnts,_)=cv2.findContours(thresh_delta.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    for cont in cnts:
        if cv2.contourArea(cont)<10000:
            
            continue
        stat=1
        
        (x,y,h,w)=cv2.boundingRect(cont)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
    
    status_list.append(stat)
    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.now())
    
    if status_list[-1]==0 and status_list[-2]==1:
        times.append(datetime.now())





    cv2.imshow('capturing',gray)
    cv2.imshow('differenvce',differ_image)
    cv2.imshow('thresh',thresh_delta)
    cv2.imshow('color',frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        if stat ==1:
            times.append(datetime.now())
        break
    
print(status_list)   
print(times)
for i in range(0,len(times),2):
    df=df.append({"start":times[i],"end":times[i+1]},ignore_index=True)
df.to_csv("times.csv")#we are exporting the data frame to the csv files
cap.release()
cv2.destroyAllWindows