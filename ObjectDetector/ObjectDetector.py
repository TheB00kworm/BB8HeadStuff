import cv2
import numpy as np

#Choose Video Camera to use
    
cap = cv2.VideoCapture(1)

#3 sets width of frames
#4 sets Height of frames
width = 1080
height = 620
cap.set(3,width)
cap.set(4,height)

#Find the center of the window
centerX = int(width/2)
centerY = int(height/2)

#Threshold to detect objects
thres = 0.45
nms_threshold = 0.2

#Obtain data to identify objects
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

#Create detection model from network using Binary file for trained weights and text file for network configuration
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

try:
    while True:
        #Sets up the camera to read the images
        ret, img = cap.read()

        #Sets up IDs, confidences level and boundary box variables for finding items
        classIds, confs, bbox = net.detect(img, confThreshold=thres)  
        bbox = list(bbox)
        confs = list(np.array(confs).reshape(1,-1)[0])
        confs = list(map(float,confs))

        #Sets up cleaning up bboxes for less overlap (non-maximum suppression)
        idxs = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

        #Creates a more stable box around objects and states what it is
        for i in idxs:
            box = bbox[i]
            confidence = confs[i]
            x,y,w,h = box[0], box[1], box[2], box[3]
            cv2.rectangle(img,(x,y), (x+w, y+h), color=(255,165,0), thickness=2)

            #Find center of bbox 
            x2 = x + int(w/2)
            y2 = y + int(h/2)
            cv2.circle(img,(x2,y2),2,(0,255,0),-1)
            loc = "x: " + str(x2) + ", y: " + str(y2)
            cv2.putText(img,loc,(x2 - 10, y2 - 10), cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),2)

            #Puts dot on center of screen
            cv2.circle(img,(centerX,centerY),4,(0,255,0),-1)
            relX = abs(x2 - centerX)
            relY = abs(y2 - centerY)

            cv2.putText(img,classNames[classIds[i]-1].upper(),(box[0]+10,box[1]+30), cv2.FONT_HERSHEY_COMPLEX,0.75,(255,165,0),2)
            cv2.putText(img,str(round(confidence*100,2)),(box[0]+10, box[1]+60), cv2.FONT_HERSHEY_COMPLEX,0.75,(255,165,0),2)

        #Show the Output Window
        cv2.imshow("Vision Output", img)
        cv2.waitKey(1)

except Exception as e: print(e)