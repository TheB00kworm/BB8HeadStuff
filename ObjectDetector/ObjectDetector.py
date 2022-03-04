import cv2
import numpy as np
from DetectObjects import detectObjects
from PictureDetects import pictureDetects
from TrackObjects import trackObjects
from FindFPS import findFPS

def main():
    #Obtain FPS of camera
    #findFPS()
    
    #Choose Video Camera to use
    cap = cv2.VideoCapture(1) #cv2.CAP_DSHOW gets rid of sync error on Windows

    #3 sets width of frames
    #4 sets Height of frames
    width = 640
    height = 480
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
    
    while True:
        action = input("\nPlease type one of the following, then press 'Enter':"
                       "\n'o' for Observation Mode"
                       "\n't' for Tracking Mode"
                       #"\n'p' for Picture stuff" 
                       "\n'end' to quit \n")
                
        if(action == "o"):
            print("Press Q to end Observation Mode")
            detectObjects(cap, net, thres, nms_threshold, classNames)
            continue
        elif(action == "t"):
            print("Press Q to end Tracking Mode")
            trackObjects(cap, net, thres, nms_threshold, classNames, centerX, centerY)
            continue
        # elif(action == 'p'):
        #     pictureDetects(net, thres, nms_threshold, classNames)
        #     break
        elif(action == "end"):
            cap.release()
            cv2.destroyAllWindows()
            break
        else:
            print("Invalid Input")

        


if __name__ == "__main__":
    main()