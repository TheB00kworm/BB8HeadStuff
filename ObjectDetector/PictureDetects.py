def pictureDetects():
    import cv2
    import numpy as np
    
    img = cv2.imread('personAtDesk.png')
    
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
                cv2.putText(img,classNames[classIds[i]-1].upper(),(box[0]+10,box[1]+30), cv2.FONT_HERSHEY_COMPLEX,0.75,(255,165,0),2)
                cv2.putText(img,str(round(confidence*100,2)),(box[0]+10, box[1]+60), cv2.FONT_HERSHEY_COMPLEX,0.75,(255,165,0),2)

            #Show the Output Window
            cv2.imshow("Vision Output", img)
            #cv2.waitKey(0)
            #Hit 'q' on the keyboard to quit
            if cv2.waitKey(0) & 0xFF == ord('q'):
                img.release()
                cv2.destroyAllWindows()
                break

    except Exception as e: print(e)


def main():
    pass
if __name__ == "__main__":
    main()