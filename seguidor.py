import math
##Realiza los procesos rastreo y conteo de los elementos en la imagen.
class seguidor:
    def __init__(self):
        self.punto_centro = {}
        self.id_count = 0

    def rastreo(self, obj):
        obj_id = []
        #obtener punto central
        for coord in obj:
            x, y, w, h = coord
            # calcula el punto medio de cada rectangulo
            centroX = int((2 * x + w) / 2)
            centroY = int((2 * y + h) / 2)
            #verifica si el objeto ya fue detectado
            obj_detect = False

            for id, pt in self.punto_centro.items():
                distancia = math.hypot(centroX - pt[0], centroY - pt[1])  # calcula la distancia
                if distancia < 45:
                    self.punto_centro[id] = (centroX, centroY)
                    obj_id.append([x, y, w, h, id])
                    obj_detect = True
                    break
            #si ahy un nuevo objeto
            if obj_detect is False:
                self.punto_centro[self.id_count] = (centroX, centroY)
                obj_id.append([x, y, w, h, self.id_count])
                self.id_count += 1

        #limpia lista
        new_punto_centro = {}
        for obj_b_id in obj_id:
            _, _, _, _, obj_idd = obj_b_id
            centro = self.punto_centro[obj_idd]
            new_punto_centro[obj_idd] = centro
        #actualizar lista
        self.punto_centro = new_punto_centro.copy()

        return obj_id



