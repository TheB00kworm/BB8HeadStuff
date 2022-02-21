import cv2
import numpy as np
from DetectObjects import detectObjects
from TrackObjects import trackObjects

def main():
    while True:
        action = input("Please type one of the following, then press 'Enter': \n'o' for Observation Mode \n't' for Tracking Mode \n'end' to quit \n")
        if(action == "o"):
            detectObjects()
            break
        elif(action == "t"):
            trackObjects()
            break
        elif(action == "end"):
            break
        else:
            print("Invalid Input")
        


if __name__ == "__main__":
    main()