import cv2
import numpy as np
import threading
from DetectObjects import detectObjects
import DetectObjects
#from PictureDetects import pictureDetects
from TrackObjects import trackObjects

#Choose Video Camera to use
DetectObjects.cap = cv2.VideoCapture(0)

visionState = 'I'

def videoGetandShow():
    while True:
        ret, DetectObjects.img = DetectObjects.cap.read()
        if ret:
            cv2.imshow("Video", DetectObjects.img)
        if cv2.waitKey(1) & threadIsActive == False:
            DetectObjects.cap.release()
            cv2.destroyAllWindows()
            break
    

def main():
    global threadIsActive
    threadIsActive = False
    
    threadVideoGetandShow = threading.Thread(target=videoGetandShow, daemon=True)
    threadIsActive = True
    threadVideoGetandShow.start()
    
    #3 sets width of frames
    #4 sets Height of frames
    width = 640
    height = 480
    # width = 1260 
    # height = 700
    DetectObjects.cap.set(3,width)
    DetectObjects.cap.set(4,height)

    #Find the center of the window
    centerX = int(width/2)
    centerY = int(height/2)

    #Threshold to detect objects
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
    #net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    #net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    net.setInputSize(320,320)
    net.setInputScale(1.0/ 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)
    
    #threadDetectObjects = threading.Thread(target=detectObjects, args=(cap, net, nms_threshold, classNames))
    
    while True:
        print(threading.active_count())
        action = input("\nPlease type one of the following, then press 'Enter':"
                       "\n'o' for Observation Mode"
                       "\n't' for Tracking Mode"
                       #"\n'p' for Picture stuff" 
                       "\n'q' to quit \n")
        # print(f"Active Threads: {threading.active_count()}")
                      
        if(action == "o"):
            print("Press Q to end Observation Mode")
            print(threading.active_count())
            detectObjects(net, nms_threshold, classNames)
            #threadDetectObjects.start()
        elif(action == "t"):
            print("Press Q to end Tracking Mode")
            trackObjects(net, nms_threshold, classNames, centerX, centerY)
            continue
        # elif(action == 'p'):
        #     pictureDetects(net, thres, nms_threshold, classNames)
        #     break
        elif(action == "q"):
            threadIsActive = False
            break
        else:
            print("\nInvalid Input")  

    

if __name__ == "__main__":
    main()