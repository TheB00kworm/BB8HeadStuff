def detectObjects(cap, net, thres, nms_threshold, classNames):
    import cv2
    import numpy as np

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
                cv2.putText(img,classNames[classIds[i]-1].upper(),(box[0]+10,box[1]+30), cv2.FONT_HERSHEY_COMPLEX,0.75,(255,165,0),2)
                cv2.putText(img,str(round(confidence*100,2)),(box[0]+10, box[1]+60), cv2.FONT_HERSHEY_COMPLEX,0.75,(255,165,0),2)

            #Show the Output Window
            cv2.imshow("Vision Output", img)
            #cv2.waitKey(1)
            # Hit 'q' on the keyboard to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # cap.release()
                cv2.destroyAllWindows()
                break

    except Exception as e: print(e)


def main():
    pass
if __name__ == "__main__":
    main()