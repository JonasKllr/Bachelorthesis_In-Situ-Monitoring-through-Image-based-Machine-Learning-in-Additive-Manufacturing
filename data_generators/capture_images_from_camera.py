#https://stackoverflow.com/questions/474528/what-is-the-best-way-to-repeatedly-execute-a-function-every-x-seconds-in-python
#https://www.geeksforgeeks.org/python-opencv-cv2-imwrite-method/

import cv2
import os 

directory = input("File directory:")
seconds_to_run = int(input("Duration in seconds:"))
os.chdir(directory)

# turn on camera. (0) for laptop cam, (2) for USB cam
cam = cv2.VideoCapture(2, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)

i = 0
while i <= seconds_to_run:
    # wait one second
    cv2.waitKey(1000)
    ret, image = cam.read()

    filename = str(i) + ".png"
    cv2.imwrite(filename, image)
    i = i+1
    
cam.release()
cv2.destroyAllWindows()
