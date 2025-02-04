#import libraries
import cv2 #opencv
import time #delay
import imutils #resize

#initailize camera
cam=cv2.VideoCapture(0)
time.sleep(1)

#initialize variables
firstFrame=None
area=500
framecount=0

while True:   #start video loop
    _,img=cam.read()
    text="Normal"  #default text
    motiondetected=False  #default flag
    
    img=imutils.resize(img,width=800)   #preprocess image 
    grayImg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #The original image is in color (BGR channels), but for motion detection, we only need intensity values (brightness) to detect changes between frames. Converting to grayscale reduces the complexity of the image
    gaussianImg=cv2.GaussianBlur(grayImg,(21,21),0)#Gaussian blur helps to reduce these high-frequency noise components by smoothing out the image. 
    
    if firstFrame is None: #initialize first frame
        firstFrame=gaussianImg
        framecount=0
        
    framecount+=1 #increment frame count
    
    imgDiff=cv2.absdiff(firstFrame,gaussianImg) #calculate differnce between frames
    
    threshImg=cv2.threshold(imgDiff,25,255,cv2.THRESH_BINARY)[1] #applying threshing..binary image where white areas represent motion, and black areas represent no motion.
    threshImg=cv2.dilate(threshImg,None,iterations=2) #applying dilation..The cv2.dilate() function takes the binary image (threshImg) and "expands" the white areas by a certain number of iterations.
    
    cnts=cv2.findContours(threshImg.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #find contours
    cnts=imutils.grab_contours(cnts)
    
    for c in cnts:  #loop through contours
        if cv2.contourArea(c)<area:
            continue
        (x,y,w,h)=cv2.boundingRect(c)
        
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),5)
        motiondetected=True
        if motiondetected:
            text="moving object detected"
        
    cv2.putText(img,text,(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2) #display text on image
    cv2.imshow("camera feed",img) #show camera feed
        
    key=cv2.waitKey(10)
    if key==ord("q"):   #exits on clicking "q"
        break
        
cam.release()
cv2.destroyAllWindows()
        