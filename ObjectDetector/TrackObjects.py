import cv2
import numpy as np
import socket
import DetectObjects
import keyboard

def trackObjects(net, nms_threshold, classNames, centerX, centerY):
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
            # src = cv2.cuda_GpuMat()
            # src.upload(img)

            #Sets up IDs, confidences level and boundary box variables for finding items
            classIds, confs, bbox = net.detect(DetectObjects.img, confThreshold=thres)  
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
                    x,y,w,h = box[0], box[1], box[2], box[3]
                    
                    #Find center of bbox
                    x2 = x + int(w/2)
                    y2 = y + int(h/2)
                    
                    #Find difference between center of screen and current position
                    relX = x2 - centerX
                    relY = y2 - centerY 
                    
                    if(abs(relX) <= curr + 15):
                        curr = abs(relX)
                        cv2.rectangle(DetectObjects.img,(x,y), (x+w, y+h), color=(255,165,0), thickness=2)
                        cv2.putText(DetectObjects.img,name,(box[0]+10,box[1]+30), cv2.FONT_HERSHEY_COMPLEX,0.75,(255,165,0),2)
                        print(classNames[classIds[i]-1].upper())
                print("NO PERSON")   
            
            sending_string = str(relX)
            #sending_string = str(relY)
            s.sendall(bytes(sending_string, 'utf-8'))

            # Hit 'q' on the keyboard to quit
            if keyboard.is_pressed('q'):
                return

    except Exception as e: print(e)

def main():
    pass
if __name__ == "__main__":
    main()