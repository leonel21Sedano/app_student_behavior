from ultralytics import YOLO
import cv2

def detectar_estudiantes(ruta_imagen):
    modelo = YOLO('best.pt')
    
    imagen = cv2.imread(ruta_imagen)
    
    resultados = modelo(imagen, conf=0.25)
    
    imagen_anotada = resultados[0].plot()
    
    detecciones = resultados[0].boxes.data.cpu().numpy()
    cantidad_estudiantes = len(detecciones)
    
    print(f"Estudiantes detectados: {cantidad_estudiantes}")
    
    cv2.imshow("Detección de Estudiantes", imagen_anotada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imwrite("estudiantes_detectados.jpg", imagen_anotada)

detectar_estudiantes(" ") 