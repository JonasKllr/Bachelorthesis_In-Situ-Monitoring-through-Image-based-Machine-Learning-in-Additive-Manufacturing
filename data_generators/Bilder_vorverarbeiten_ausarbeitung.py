import cv2
import os, os.path

#Ordner in dem Bilder sind, die Bearbeitet werden sollen
directory = r"C:\Users\Jonas\Documents\Studium\Bachelor\Bachelorarbeit\Skripte_Ausarbeitung\Bilder\Wuerfel_2\Original"
content = os.listdir(directory)

for i in range(0, len(content)):
    os.chdir(directory)

    #Bild schneiden
    img = cv2.imread(str(content[i]))
    crop_img = img[490:630, 925:1235]       #aktuell: [490:630, 925:1235]

    os.chdir(r"C:\Users\Jonas\Documents\Studium\Bachelor\Bachelorarbeit\Skripte_Ausarbeitung\Bilder\Wuerfel_2\Bearbeitet")
    cv2.imwrite("geschnitten.png", crop_img)

    #Bild drehen
    rot_img = cv2.rotate(crop_img, cv2.ROTATE_180)
    cv2.imwrite("gedreht.png", rot_img)

    #Graustufen
    grey_img = cv2.cvtColor(rot_img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("grau.png", grey_img)

    #minimalen/maximalen Grauwert ermitteln
    minimum = 127
    maximum = 127

    for x in range(0, grey_img.shape[1]):
        for y in range(0, grey_img.shape[0]):
            if grey_img[y][x] < minimum:
                minimum = grey_img[y][x]
            elif grey_img[y][x] > maximum:
                maximum = grey_img[y][x]

    #Grauwerte anpassen
    schwarz = 0
    weiß = 255

    faktor = (weiß - schwarz) / (maximum - minimum)

    for x in range(0, grey_img.shape[1]):
        for y in range(0, grey_img.shape[0]):
            grey_img[y][x] = (grey_img[y][x]-minimum) * faktor
    
    #Bilder speichern
    cv2.imwrite("manipuliert.png", grey_img)