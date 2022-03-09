def trackObjects(cap, net, thres, nms_threshold, classNames, centerX, centerY):
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
            curr = 320
            
            #Creates a more stable box around objects and states what it is and how confidant it is
            for i in idxs:
                name = classNames[classIds[i]-1].upper()
                
                #Checks to see if Object is a person before creating box around it
                if(name == "PERSON"):
                    box = bbox[i]
                    confidence = confs[i]
                    
                    x,y,w,h = box[0], box[1], box[2], box[3]
                    
                    #Find center of bbox
                    x2 = x + int(w/2)
                    y2 = y + int(h/2)
                    
                    #Find difference between center of screen and current position
                    relX = abs(x2 - centerX)
                    relY = abs(y2 - centerY)

                    if(relX <= curr - 10):
                        curr = relX
                        cv2.rectangle(img,(x,y), (x+w, y+h), color=(255,165,0), thickness=2)
                        cv2.putText(img,name,(box[0]+10,box[1]+30), cv2.FONT_HERSHEY_COMPLEX,0.75,(255,165,0),2)
                        cv2.putText(img,str(round(confidence*100,2)),(box[0]+10, box[1]+60), cv2.FONT_HERSHEY_COMPLEX,0.75,(255,165,0),2)

                        #Find center of bbox and place coordinates
                        cv2.circle(img,(x2,y2),2,(0,255,0),-1)
                        loc = "x: " + str(x2) + ", y: " + str(y2)
                        cv2.putText(img,loc,(x2 - 10, y2 - 10), cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),2)

                        #Puts dot on center of screen
                        #cv2.circle(img,(centerX,centerY),4,(0,255,0),-1)
                        
                        #WIll Shift motors to center object
                        #Currently Marks the center of Frame relative to Object
                        if(x2 >= centerX & y2 >= centerY):
                            cv2.circle(img,(x2-relX,y2-relY),4,(255,165,0),-1)
                        elif(x2 > centerX & y2 < centerY):
                            cv2.circle(img,(x2-relX,y2+relY),4,(255,165,0),-1)
                        elif(x2 < centerX & y2 > centerY):
                            cv2.circle(img,(x2+relX,y2-relY),4,(255,165,0),-1)
                        else:
                            cv2.circle(img,(x2+relX,y2+relY),4,(255,165,0),-1)

            #Show the Output Window
            cv2.imshow("Tracking", img)
            
            #cv2.waitKey(1)
            # Hit 'q' on the keyboard to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                #cap.release()
                cv2.destroyAllWindows()
                break

    except Exception as e: print(e)


def main():
    pass
if __name__ == "__main__":
    main()