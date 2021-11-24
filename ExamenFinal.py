from seguidor import *
import cv2
import numpy as np


def tomarPunto(idImg, image_draw, colorPunto):
    points = []
    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
    cv2.namedWindow(idImg)
    cv2.setMouseCallback(idImg, click)
    points1 = []
    point_counter = 0
    while True:
        cv2.imshow(idImg, image_draw)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("x"):
            points1 = points.copy()
            points = []
            break
        if len(points) > point_counter:
            point_counter = len(points)
            cv2.circle(image_draw, (points[-1][0], points[-1][1]), 3, colorPunto, -1)
            print(points)
    cv2.destroyWindow(idImg) #una vez selecionados los puntos cierra la imagen
    return points1 # retorna los puntos




tracking = seguidor()
red = [0, 0, 255] #color para pintar puntos

img1 = cv2.imread("soccer_game.png")#Imagen de entrada
img2 = img1
img3 = img2
img4 = img3
img5 = img4
img6 = img5
img7 = img6
img8 = img7
img9 = img8
img10 = img9

height, width  = img10.shape[:2]
video = cv2.VideoWriter('video.wmv',cv2.VideoWriter_fourcc(*'mp4v'),2,(width,height))
video.write(img1)
video.write(img2)
video.write(img3)
video.write(img4)
video.write(img5)
video.write(img6)
video.write(img7)
video.write(img8)
video.write(img9)
video.write(img10)

#liberar recursos
video.release()

#Punto 1 _ Examen final

gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
matrix = img1[:,:,1]
TotalPixeles = matrix.size



#Propiedades de la foto
histograma = cv2.calcHist([matrix], [0], None, [256], [0, 256])
plt.plot(histograma, color='gray')
plt.xlim([0, 256])
plt.show()
plt.imshow(img1)
plt.title('Imagen BGR')
plt.show()


number_of_white_pix = np.sum(matrix < 80)
number_of_white_pix2 = np.sum(matrix > 120)
#number_of_black_pix = np.sum(matrix == 0)
Total_Cesped =  (TotalPixeles - (number_of_white_pix + number_of_white_pix))/TotalPixeles
print('% Pixeles Cesped:', Total_Cesped)
print('Total pixeles Imagen:', TotalPixeles)

matrix[matrix > 120] = 255
matrix[matrix < 80] = 255
cv2.imshow('Cesped', matrix)
cv2.waitKey(0)

print("fin proceso")

#Tomando la imagen de grises BGR2GRAY, se extraen los pixeles blancos y negros ==> Sumatoria

t, dst = cv2.threshold(matrix, 120, 250, cv2.THRESH_BINARY)

cv2.imshow('umbral', gray)
cv2.imshow('result', dst)
cv2.waitKey(0)

#Punto 2

cap = cv2.VideoCapture("video.wmv")#video entrada

obj_detec = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=50)


ret, frame = cap.read()
punto1 = tomarPunto("Primer frame", frame, red) #llama al metodo que pinta rojo
punto = np.array(punto1) #pasa los fumtos a un array para el metodo cv2.drawContours
# coordenada para realizar el conteo
line_goal = [ (punto[2][0], punto[2][1]), (punto[3][0],punto[3][1]) ]

count =0

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
while True:
    ret, frame = cap.read()
    alto, ancho, _ = frame.shape

    cv2.drawContours(frame, [punto], -1, (0, 255, 0), 2)# dibuja los puntos capturados con click

    imgMask = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)#imagen para usar de mascara
    imgMask = cv2.drawContours(imgMask, [punto], -1, 255, -1)# dibuja en blanco el area del polygono que encierra los puntos
    imgMaskArea = cv2.bitwise_and(frame, frame, mask=imgMask)# muestra sobre la imagen principal solo lo que esta en la mascara


    mask = obj_detec.apply(imgMaskArea) #aplica el sustractor de fondo sobre la imagen mascara
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY) #elimina las sombras despues de aplicar el sustractor de fondo

    img_dilat = cv2.dilate(mask, np.ones((7, 7)))
    img_dilat = cv2.morphologyEx(img_dilat, cv2.MORPH_ELLIPSE, kernel)
    img_dilat = cv2.morphologyEx(img_dilat, cv2.MORPH_ELLIPSE, kernel)

    contornos, _ = cv2.findContours(img_dilat, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#encuentra los contornos
    cont_detect = [] #variable para guardar  cada contorno
    for cont in contornos:
        area = cv2.contourArea(cont)#calcula el area de cada contorno

        if area > 50: #visualiza los contornos con areas mayores
            x, y, w, h = cv2.boundingRect(cont)# extrae los puntos del rectangulo de cada contorno
            cont_detect.append([x, y, w, h]) #Guarda los puntos
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)# dibuja rectangulos sobre cada contorno, color Rojo

    #Tracking
    infor_id = tracking.rastreo(cont_detect) #recibe todos los contornos de cada frame

    for info in infor_id:
        x, y, w, h, ID = info
        cX = int((2 * x + w) / 2)
        cY = int((2 * y + h) / 2)
        cv2.circle(frame, (cX, cY), 5, red, -1)  # dibuja un punto sobre cada objeto
        #dibuja el id-Contador
        cv2.putText(frame, str(ID), (x, y-15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)


    cv2.imshow("mask", mask)
    cv2.imshow("frame", frame)


    key = cv2.waitKey(20)
    if key ==27:
        break

cap.release()
cv2.destroyAllWindows()

# Punto 3

def tomarPunto(idImg, image_draw, colorPunto):
    points = []
    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
    cv2.namedWindow(idImg)
    cv2.setMouseCallback(idImg, click)
    points1 = []
    point_counter = 0
    while True:
        cv2.imshow(idImg, image_draw)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("x"):
            points1 = points.copy()
            points = []
            break
        if len(points) > point_counter:
            point_counter = len(points)
            cv2.circle(image_draw, (points[-1][0], points[-1][1]), 3, colorPunto, -1)
            print(points)
    cv2.destroyWindow(idImg) #una vez selecionados los puntos cierra la imagen
    return points1 # retorna los puntos