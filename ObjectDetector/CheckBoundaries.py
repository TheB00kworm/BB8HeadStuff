
def checkBoundaries(zpan,xtilt, ytilt):
    #2.9in Forward/Backward
    #3.5in Left/Right
    #if out a range, set pan and/or tilt to max
    
    if ytilt > 14:
        ytilt = 14
    if ytilt < 0:
        ytilt = 0
    if xtilt > 17:
        xtilt = 17
    if xtilt < 0:
        xtilt = 0
    if zpan > 90:
        zpan = 90
    if zpan < 0:
        zpan = 0
    
    return(zpan, xtilt, ytilt)

def main():
    pass
if __name__ == "__main__":
    main()