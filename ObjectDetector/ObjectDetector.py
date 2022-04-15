import time
import cv2
import numpy as np
import threading
import socket
# from DetectObjects import detectObjects
# import DetectObjects
# #from PictureDetects import pictureDetects
# from TrackObjects import trackObjects

global cap
global img
global visionState

#Choose Video Camera to use
cap = cv2.VideoCapture(0)

visionState = 'v'

def videoGetandShow():
    global img, visionState, semState
    try:
        var = 0
        while True:
            if threadIsActive:
                var = 0
                ret,img = cap.read()
                cv2.imshow("Video", img)
            else:
                if visionState == 'o' and var == 0:
                    cv2.destroyWindow('Video')
                    var = 1
                continue
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                    
    except Exception as e: print(e)
    # finally:
    #     cv2.destroyAllWindows()

def main():
    global threadIsActive
    threadIsActive = False

    # cap = cv2.VideoCapture(0)

    #3 sets width of frames
    #4 sets Height of frames
    width = 640
    height = 480
    # width = 1260
    # height = 700

    cap.set(3,width)
    cap.set(4,height)

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
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    net.setInputSize(320,320)
    net.setInputScale(1.0/ 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    #threadDetectObjects = threading.Thread(target=detectObjects, args=(sem,net, nms_threshold, classNames), daemon=True)
    threadVideoGetandShow = threading.Thread(target=videoGetandShow, daemon=True)
    threadIsActive = True
    threadVideoGetandShow.start()

    global visionState
    while True:
        visionState = 'v'
        action = input("\nPlease type one of the following, then press 'Enter':"
                       "\n'o' for Observation Mode"
                       "\n't' for Tracking Mode"
                       #"\n'p' for Picture stuff"
                       "\n'q' to quit \n")

        if(action == "o"):
            threadIsActive = False
            print("Press E to end Observation Mode")
            threadIsActive = False
            time.sleep(0.05)
            visionState = 'o'
            detectObjects(net, nms_threshold, classNames)
        elif(action == "t"):
            print("Press E to end Tracking Mode")
            visionState = 't'
            trackObjects(net, nms_threshold, classNames, centerX, centerY)
            continue
        # elif(action == 'p'):
        #     pictureDetects(net, thres, nms_threshold, classNames)
        #     break
        elif(action == "q"):
            threadIsActive = False
            cap.release()
            cv2.destroyAllWindows()
            break
        else:
            print("\nInvalid Input")
    threadVideoGetandShow.join()



####################################################################################################
######################################### DETECTING OBJECTS ########################################
####################################################################################################
def detectObjects(net, nms_threshold, classNames):
    global img
    global threadIsActive
    global visionState

    try:
        thres = 0.50

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
            for i in range(0, len(idxs)):
                box = bbox[i]
                confidence = confs[i]
                x,y,w,h = box[0], box[1], box[2], box[3]
                cv2.rectangle(img,(x,y), (x+w, y+h), color=(255,165,0), thickness=2)
                cv2.putText(img,classNames[classIds[i][0]-1].upper(),(box[0]+10,box[1]+30), cv2.FONT_HERSHEY_COMPLEX,0.75,(255,165,0),2)
                cv2.putText(img,str(round(confidence*100,2)),(box[0]+10, box[1]+60), cv2.FONT_HERSHEY_COMPLEX,0.75,(255,165,0),2)
            #Show the Output Window
            cv2.imshow("Observing", img)
            #cv2.waitKey(1)
            # Hit 'q' on the keyboard to quit
            if (cv2.waitKey(1) & 0xFF == ord('e')) or visionState != 'o':
                visionState = 'i'
                break

    except Exception as e: print(e)
    finally:
        threadIsActive = True
        cv2.destroyWindow('Observing')


####################################################################################################
#################################### TRACKING CENTERMOST PERSON ####################################
####################################################################################################
def trackObjects(net, nms_threshold, classNames, centerX, centerY):
    global img

    port = 123
    host = '10.227.71.218'
    addressPort = host,port

    relX = centerX
    relY = centerY
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(addressPort)
        thres = 0.40
        curr = centerX

        while True:
            #Sets up the camera to read the images
            # ret, img = cap.read()

            #Sets up IDs, confidences level and boundary box variables for finding items
            classIds, confs, bbox = net.detect(img, confThreshold=thres)
            bbox = list(bbox)
            confs = list(np.array(confs).reshape(1,-1)[0])
            confs = list(map(float,confs))

            #Sets up cleaning up bboxes for less overlap (non-maximum suppression)
            idxs = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

            #Puts dot on center of screen
            # cv2.circle(img,(centerX,centerY),4,(0,255,0),-1)

            #Creates a more stable box around objects and states what it is and how confidant it is
            for i in range(0, len(idxs)):
                name = classNames[classIds[i][0]-1].upper()

                #Checks to see if Object is a person before creating box around it
                if(name == "PERSON"):
                    box = bbox[i]
                    x,y,w,h = box[0], box[1], box[2], box[3]

                    #Find center of bbox
                    x2 = x + int(w/2)
                    y2 = y + int(h/2)

                    #Find difference between center of screen and current position
                    relX = x2 - centerX
                    relY = y2 - centerY

                    if(abs(relX) <= curr + 15):
                        curr = abs(relX)
                        # cv2.rectangle(img,(x,y), (x+w, y+h), color=(255,165,0), thickness=2)
                        # cv2.putText(img,name,(box[0]+10,box[1]+30), cv2.FONT_HERSHEY_COMPLEX,0.75,(255,165,0),2)
                        print(name)
                    else:
                        print("NO PERSON")

            #Show the Output Window
            # cv2.imshow("Tracking", img)

            sending_string = str(relX)
            #sending_string = str(relY)
            s.sendall(bytes(sending_string, 'utf-8'))

            # Hit 'q' on the keyboard to quit
            if cv2.waitKey(1) & 0xFF == ord('e'):
                break

    except Exception as e: print(e)






if __name__ == "__main__":
    main()