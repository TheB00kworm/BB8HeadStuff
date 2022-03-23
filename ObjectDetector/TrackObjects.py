import cv2
import numpy as np
from CheckBoundaries import checkBoundaries

def trackObjects(cap, net, nms_threshold, classNames, centerX, centerY, zpan, xtilt, ytilt):
    try:
        thres = 0.40
        curr = centerX
        
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
            
            #Puts dot on center of screen
            # cv2.circle(img,(centerX,centerY),4,(0,255,0),-1)
            
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
                    relX = x2 - centerX
                    relY = y2 - centerY

                    if(abs(relX) <= curr + 15):
                        curr = relX
                        cv2.rectangle(img,(x,y), (x+w, y+h), color=(255,165,0), thickness=2)
                        cv2.putText(img,name,(box[0]+10,box[1]+30), cv2.FONT_HERSHEY_COMPLEX,0.75,(255,165,0),2)
                        #cv2.putText(img,str(round(confidence*100,2)),(box[0]+10, box[1]+60), cv2.FONT_HERSHEY_COMPLEX,0.75,(255,165,0),2)

                        #Find center of bbox and place coordinates
                        # cv2.circle(img,(x2,y2),2,(0,255,0),-1)
                        # loc = "x: " + str(x2) + ", y: " + str(y2)
                        # cv2.putText(img,loc,(x2 - 10, y2 - 10), cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),2)
                        
                        #Will Shift Servos to center object
                        #Need some type of conversion from pixel to angle... guessing here with 75
                        if abs(relX) > 15:
                            #if zpan > some number that requires bb8 to turn it's body
                                #xtilt = xtilt-relx/75
                            zpan = zpan-relX/75
                            
                        if abs(relY) > 15:
                            ytilt = ytilt-relY/75
                            
                            
                        #check to see if within range (FIND SERVO RANGE)
                        #checkBoundaries(zpan,ytilt,xtilt)                     
                        
                        #Convert to string to send to Pi
                        zpan_string = str(zpan)
                        ytilt_string = str(ytilt)
                        xtilt_string = str(xtilt)
                        #return(zpan_string,ytilt_string,xtilt_string)
                        
                        #rotate left/right Servo.angle=zpan
                        #UP/Down Servo.angle=ytilt
                        
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