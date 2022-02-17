import cv2
import numpy as np


#Test an Image for detection
#img = cv2.imread('Angel.jpg')

#Choose Video Camera to use
    #3 sets width of frames
    #4 sets Height of frames
    #10 sets brightness
cap = cv2.VideoCapture(1)
#cap.set(3,1280) 
#cap.set(4,720)
#cap.set(10,70)

#Threshold to detect objects
thres = 0.45
nms_threshold = 0.2

#Obtain data to identify objects
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
#print(classNames)

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

while True:
    #Sets up the camera to read the images
    success, img = cap.read()

    #Sets up IDs, confidences level and boundary box variables for finding items
    classIds, confs, bbox = net.detect(img, confThreshold=thres)  
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1,-1)[0])
    confs = list(map(float,confs))
    #print(type(bbox))
    #print(type(confs))
    #print(confs)

    #Sets up cleaning up bboxes for less overlap (non-maximum suppression)
    idxs = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)
    #print(idxs)

    #Creates a more stable box around objects and states what it is
    for i in idxs:
        box = bbox[i]
        confidence = confs[i]
        x,y,w,h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img,(x,y), (x+w, y+h), color=(255,165,0), thickness=2)
        cv2.putText(img,classNames[classIds[i]-1].upper(),(box[0]+10,box[1]+30), cv2.FONT_HERSHEY_COMPLEX,1,(255,165,0),2)
        cv2.putText(img,str(round(confidence*100,2)),(box[0]+10, box[1]+60), cv2.FONT_HERSHEY_COMPLEX,1,(255,165,0),2)

    #CREATES TOO MANY BOXES AND CONFUSION
    ##Create box around object, state what it is, and the accuracy to which it knows it is
    #if len(classIds) !=0:
    #    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
    #        cv2.rectangle(img,box,color=(255,165,0),thickness=2)
    #        cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30), cv2.FONT_HERSHEY_COMPLEX,1,(255,165,0),2)
    #        cv2.putText(img,str(round(confidence*100,2)),(box[0]+200, box[1]+30), cv2.FONT_HERSHEY_COMPLEX,1,(255,165,0),2)

    #Show the Output Window
    cv2.imshow("Vision Output", img)
    cv2.waitKey(1)