import cv2
import os

directory = r"C:\PATH\TO\ORIGINAL\IMAGES"
content = os.listdir(directory)

for i in range(0, len(content)):
    os.chdir(directory)

    # crop image
    img = cv2.imread(str(content[i]))
    crop_img = img[490:630, 925:1235]

    os.chdir(r"C:\PATH\TO\SAVE\TO")
    cv2.imwrite("cropped.png", crop_img)

    # rotate image
    rot_img = cv2.rotate(crop_img, cv2.ROTATE_180)
    cv2.imwrite("rotated.png", rot_img)

    # gray scale
    grey_img = cv2.cvtColor(rot_img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("grey.png", grey_img)

    # find min/max gray scale value
    minimum = 127
    maximum = 127

    for x in range(0, grey_img.shape[1]):
        for y in range(0, grey_img.shape[0]):
            if grey_img[y][x] < minimum:
                minimum = grey_img[y][x]
            elif grey_img[y][x] > maximum:
                maximum = grey_img[y][x]

    # adapt grey scale values
    black = 0
    white = 255
    factor = (white - black) / (maximum - minimum)

    for x in range(0, grey_img.shape[1]):
        for y in range(0, grey_img.shape[0]):
            grey_img[y][x] = (grey_img[y][x]-minimum) * factor
    
    cv2.imwrite("manipuliert.png", grey_img)