import cv2
import os

directory = r"C:\PATH\TO\ORIGINAL\IMAGES"
content = os.listdir(directory)

for i in range(0, len(content)):
    os.chdir(directory)
    
    #Bild schneiden
    img = cv2.imread(str(content[i]))
    img = img[490:630, 925:1235]

    # gray scale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # rotate image
    img = cv2.rotate(img, cv2.ROTATE_180)

    # find min/max gray scale value
    minimum = 127
    maximum = 127

    for x in range(0, img.shape[1]):
        for y in range(0, img.shape[0]):
            if img[y][x] < minimum:
                minimum = img[y][x]
            elif img[y][x] > maximum:
                maximum = img[y][x]

    #adapt grey scale values
    black = 0
    white = 255

    factor = (white - black) / (maximum - minimum)

    for x in range(0, img.shape[1]):
        for y in range(0, img.shape[0]):
            img[y][x] = (img[y][x]-minimum) * factor
    
    os.chdir(r"C:\PATH\TO\SAVE\TO")
    cv2.imwrite("Grauw_angegl_" + content[i], img)