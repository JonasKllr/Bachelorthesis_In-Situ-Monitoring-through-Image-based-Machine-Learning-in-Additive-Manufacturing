import cv2
import os, os.path

#Ordner in dem Bilder sind, die Bearbeitet werden sollen
directory = r"C:\Users\Jonas\Documents\Studium\Bachelor\Bachelorarbeit\3D_Druck\Bilder\Datensatz_raw\Original\Wuerfel_layershift_vorne_16"
content = os.listdir(directory)

for i in range(0, len(content)):
    os.chdir(directory)

    # Bild einlesen
    img = cv2.imread(str(content[i]))
    
    #Bild schneiden
    img = img[490:630, 925:1235]       #aktuell: [490:630, 925:1235]

    # Graustufen
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Bild drehen
    img = cv2.rotate(img, cv2.ROTATE_180)

    #minimalen/maximalen Grauwert ermitteln
    minimum = 127
    maximum = 127

    for x in range(0, img.shape[1]):
        for y in range(0, img.shape[0]):
            if img[y][x] < minimum:
                minimum = img[y][x]
            elif img[y][x] > maximum:
                maximum = img[y][x]

    #Grauwerte anpassen
    schwarz = 0
    weiß = 255

    faktor = (weiß - schwarz) / (maximum - minimum)

    for x in range(0, img.shape[1]):
        for y in range(0, img.shape[0]):
            img[y][x] = (img[y][x]-minimum) * faktor
    
    #Bilder speichern
    os.chdir(r"C:\Users\Jonas\Documents\Studium\Bachelor\Bachelorarbeit\3D_Druck\Bilder\Datensatz_raw\Bearbeitet\Wuerfel_layershift_vorne_16_bearbeitet")
    cv2.imwrite("Grauw_angegl_" + content[i], img)