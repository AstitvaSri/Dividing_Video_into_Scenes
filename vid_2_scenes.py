import cv2
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import numpy as np
import sys


def check_shot_boundary(frame_1,frame_2):
    
    frame1 = np.copy(frame_1)
    frame2 = np.copy(frame_2)
    
    frame1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2HSV)
    frame2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2HSV)
    
    hist1_H = cv2.calcHist([frame1], [0], None, [8],[0,256])
    hist2_H = cv2.calcHist([frame2], [0], None, [8],[0,256])
    hist1_S = cv2.calcHist([frame1], [1], None, [4],[0,256])
    hist2_S = cv2.calcHist([frame2], [1], None, [4],[0,256])
    hist1_V = cv2.calcHist([frame1], [2], None, [4],[0,256])
    hist2_V = cv2.calcHist([frame2], [2], None, [4],[0,256])
    
    D = []
    for i in range(len(hist1_H)):
        D.append(min(hist1_H[i],hist2_H[i]))
    for i in range(len(hist1_S)):        
        D.append(min(hist1_S[i],hist2_S[i]))
    for i in range(len(hist1_V)):        
        D.append(min(hist1_V[i],hist2_V[i]))
    D = np.array(D)
    return np.sum(D)



#detecting potential boundaries
print("Analysing HSV histogram of the frames.....")
captured = cv2.VideoCapture("inputs/aquaman.mp4")
count=1
frame_prev = None
frame_curr = None
D_dict = dict()
shot_boundaries = []
while captured.isOpened():
    ret,frame = captured.read()
    if ret==False:
        break
        
        
    FPS = 29    
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')     
    if int(major_ver)  < 3 :
        FPS = captured.get(cv2.cv.CV_CAP_PROP_FPS)
    else :
        FPS = captured.get(cv2.CAP_PROP_FPS) 
        
        
    cv2.imwrite("inputs/#frames/frame"+str(count)+".jpg", frame)
    frame_curr = frame
    if count>1:
        D = check_shot_boundary(frame_prev,frame_curr)
        D_dict[count] = D
    frame_prev = frame_curr
    count += 1

print("Detecting potential shot boundaries.....")    
shot_boundaries = [1]
l = np.array(list(D_dict.values()))
thresh = np.max(l)/1.7
for key in D_dict.keys():
    if D_dict.get(key) < thresh:
        shot_boundaries.append(key)
        

#detecting key frames in each shot
print("Determining key frames.....")    
starts = [i for i in shot_boundaries]
ends = [i-1 for i in shot_boundaries[1:]]
shots = list(zip(starts,ends))
KEY__FRAMES = []
for start,end in shots:
    if abs(start-end)>2:
        middle = (start+end) // 2
        key_frames = list()
        key_frames.append(middle)        
        for ind in range(start+1,end):
            f_img = cv2.imread("inputs/#frames/frame"+str(ind)+".jpg")
            maxDk = -1
            for k in key_frames:
                k_img = cv2.imread("inputs/#frames/frame"+str(k)+".jpg")
                maxDk = max(maxDk , check_shot_boundary(f_img,k_img) )
            if maxDk<thresh:
                key_frames.append(ind)
        KEY__FRAMES.append(np.array([start,key_frames,end]))     

        
#finding shot coherence
print("Finding coherence among shots.....")
window_size = 5
NEW_SHOT_BOUNDARIES = dict()
NEW_SHOT_BOUNDARIES.clear()
for i in range(window_size-1,len(KEY__FRAMES)):
    NEW_SHOT_BOUNDARIES[i] = "Yes"
    coherence = False
    matched_shot = -1
    for k_i in  KEY__FRAMES[i][1:-1][0]:        
        k_i_frame = cv2.imread("inputs/#frames/frame"+str(k_i)+".jpg")
        for j in range(i-(window_size-1),i):
            NEW_SHOT_BOUNDARIES[j] = "Yes"
            for k_j in KEY__FRAMES[j][1:-1][0]:
                k_j_frame = cv2.imread("inputs/#frames/frame"+str(k_j)+".jpg")
                if check_shot_boundary(k_i_frame,k_j_frame) >= thresh:
                    coherence = True
                    matched_shot = j
                    for shot_ind in range(j,i):
                        NEW_SHOT_BOUNDARIES[shot_ind]="No"
                    break
            if coherence == True:
                break
        if coherence == True:
            break
            
keys = list(NEW_SHOT_BOUNDARIES.keys())
keys.sort()
FINAL_BOUNDARIES = []
for k in keys:
    if NEW_SHOT_BOUNDARIES.get(k) == "Yes":
        FINAL_BOUNDARIES.append(shots[k][0])

x = FINAL_BOUNDARIES
y = [i-1 for i in FINAL_BOUNDARIES[1:]]
y[-1] = y[-1]+1
FINAL_SHOTS = list(zip(x,y))

#segmenting shots
print("Dividing the scenes.....")
mypath = "inputs/#frames"
frame_names = [f for f in listdir(mypath) if isfile(join(mypath, f))]
frame_names_tuples = [(strng,int(strng[5:-4])) for strng in frame_names]
frame_names_tuples.sort(key = lambda x: x[1])
frame_names_tuples

shot_num = 1
for (start,end) in FINAL_SHOTS:    
    init = False
    video = []
    for i in range(start,end):
        name = mypath+'/'+'frame'+str(i)+'.jpg'
        frame = cv2.imread(name)
        if init==False:       
            height,width,layers=frame.shape
            video=cv2.VideoWriter('outputs/Final_Shots/'+str(shot_num)+'.mkv',0,FPS,(width,height))
            init = True
        video.write(frame)
    video.release()
    shot_num+=1
print("DONE!")
