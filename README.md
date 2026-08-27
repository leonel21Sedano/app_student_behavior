# Detector de comportamientos estudiantiles

Aplicacion de escritorio para detectar comportamientos en imagenes mediante un modelo YOLO y mostrar los resultados en una interfaz grafica creada con Tkinter. El proyecto incluye deteccion sobre imagenes, deteccion en tiempo real con camara y un modulo de analisis estadistico con exportacion de datos y graficos a Excel.

La interfaz esta en espanol y el codigo utiliza nombres de variables y funciones en ingles.

## Funcionalidades

- Seleccion de imagenes en formato JPG, JPEG, PNG o BMP.
- Inferencia con un modelo YOLO entrenado para las clases del proyecto.
- Visualizacion de las cajas de deteccion sobre la imagen.
- Conteo total de detecciones.
- Deteccion en tiempo real mediante la camara predeterminada del equipo.
- Graficos de barras y de pastel por clase detectada.
- Resumen con total, promedio, clase mas frecuente y clase menos frecuente.
- Exportacion de graficos a PNG o PDF.
- Exportacion de datos y graficos a un archivo XLSX.

## Requisitos

- Windows con Python 3.10 o posterior.
- Una camara compatible con OpenCV para usar el modo de deteccion en tiempo real.
- El archivo de pesos del modelo, normalmente `best.pt`.
- Las dependencias incluidas en `requirements.txt`.

El archivo `best.pt` no se incluye en el repositorio y esta excluido mediante `.gitignore`, ya que los modelos pueden ser archivos grandes o contener material que no debe publicarse.

## Instalacion

Abre PowerShell en la carpeta del proyecto y crea un entorno virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Instala las dependencias:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Si PowerShell impide activar el entorno virtual, habilita la ejecucion de scripts para tu cuenta:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Despues, vuelve a activar `.venv` e instala las dependencias.

### Uso de GPU

La aplicacion principal inicializa el detector con `device="cpu"`. Para utilizar CUDA, instala una version compatible de `torch` y `torchvision` desde la pagina oficial de PyTorch y cambia el dispositivo en `app_deteccion_estudiantes.py`:

```python
self.yolo = YOLOModel(model_path="best.pt", device="cuda", conf=0.25)
```

La compatibilidad de CUDA depende de la tarjeta grafica, los controladores y la version instalada de PyTorch.

## Modelo

Coloca el archivo de pesos en la carpeta principal del proyecto:

```text
yolov8/
|- best.pt
|- app_deteccion_estudiantes.py
|- yolo_model.py
```

El nombre y la ruta pueden cambiarse al crear `YOLOModel`. Si el archivo no existe, la aplicacion mostrara un error al ejecutar la primera deteccion.

## Ejecucion

Con el entorno virtual activo y `best.pt` en la carpeta del proyecto, inicia la aplicacion:

```powershell
python .\app_deteccion_estudiantes.py
```

## Flujo de uso

1. Pulsa `Cargar Imagen` y selecciona una imagen.
2. Pulsa `Detectar Comportamientos` para ejecutar la inferencia.
3. Revisa la imagen anotada y el total de detecciones.
4. Pulsa `Guardar Resultado` para guardar la imagen procesada.
5. Pulsa `Analisis Estadistico` para generar los graficos y el resumen.
6. Desde la ventana estadistica, usa `Exportar Graficos` para guardar PNG, PDF o XLSX, o `Guardar en Excel` para generar directamente un archivo XLSX.
7. Pulsa `Deteccion Camara` para iniciar el modo en tiempo real. La tecla `q` tambien detiene la ventana de OpenCV.

## Estructura del proyecto

```text
.
|- app_deteccion_estudiantes.py  # Interfaz principal de Tkinter.
|- yolo_model.py                 # Wrapper con carga diferida del modelo YOLO.
|- camera_detection.py           # Captura y procesamiento de la camara.
|- analisis_estadistico.py       # Graficos, resumen y exportacion.
|- alumnos_yolo.py               # Script sencillo de prueba sobre una imagen.
|- requirements.txt              # Dependencias fijadas por version.
|- .gitignore                    # Excluye modelos, entornos y archivos generados.
|- best.pt                       # Pesos locales; no se publica.
```

La carpeta `1.23.45` contiene imagenes de apoyo del proyecto. No es necesaria para iniciar la interfaz si ya se dispone del modelo y de una imagen de entrada.

## Componentes principales

### `StudentDetectionApp`

Gestiona la ventana principal, la seleccion de archivos, la vista previa, los botones y el estado de la deteccion. `detect_students` lee la imagen con OpenCV, llama a `YOLOModel.detect_image` y conserva los resultados en `detection_data`.

### `YOLOModel`

Encapsula la instancia de Ultralytics YOLO. El modelo se carga solo cuando se necesita por primera vez. `detect_image` devuelve la imagen anotada, las cajas, los nombres de clase y el resultado original.

### `CameraDetector`

Abre la camara predeterminada con OpenCV, procesa los fotogramas en un hilo separado y muestra una ventana de deteccion en tiempo real. La camara se libera al detener el proceso o al pulsar `q`.

### `StatisticalAnalysisWindow`

Convierte las detecciones en conteos por clase, genera los graficos y permite guardar los datos junto con una imagen de los graficos en un libro de Excel.

## Formato de los resultados

La aplicacion utiliza el siguiente diccionario para compartir los resultados entre modulos:

```python
{
    "detections": numpy.ndarray | None,
    "classes": dict,
    "total_count": int,
    "image_path": str,
}
```

Cada fila de `detections` suele tener el formato de Ultralytics:

```text
[x1, y1, x2, y2, confianza, id_de_clase]
```

El analisis estadistico utiliza la columna `id_de_clase` para contar las detecciones y consulta `classes` para mostrar el nombre de cada clase.

## Archivos generados y control de versiones

El archivo `.gitignore` excluye los siguientes elementos:

- Entornos virtuales y cache de Python.
- Configuraciones de IDE.
- Pesos y binarios de modelos (`*.pt`, `*.pth`, `*.ckpt`, `*.onnx`).
- Carpetas de resultados y datos.
- Archivos Excel y registros temporales.

Antes de publicar el repositorio, verifica que no haya modelos, imagenes privadas, datos personales ni credenciales incluidos en el historial de Git.

## Solucion de problemas

### No se encuentra `best.pt`

Confirma que el archivo este en la carpeta desde la que se ejecuta el programa o actualiza `model_path` con una ruta valida.

### No se abre la camara

Comprueba que ninguna otra aplicacion este usando la camara y que Windows haya concedido permisos de acceso. Si hay varias camaras conectadas, cambia el indice `device` en `CameraDetector`.

### La ventana no muestra la imagen

Verifica que el archivo pueda abrirse normalmente y que tenga una extension compatible. OpenCV debe poder leer la imagen antes de iniciar la deteccion.

### Error al instalar PyTorch

Instala primero la variante de `torch` y `torchvision` compatible con tu sistema. Para CPU, utiliza los comandos recomendados por PyTorch y despues instala el resto de las dependencias con:

```powershell
pip install -r requirements.txt --no-deps
```

## Verificacion local

Antes de subir cambios, se recomienda comprobar la sintaxis y ejecutar las herramientas de calidad disponibles en el entorno:

```powershell
python -m compileall .\app_deteccion_estudiantes.py .\yolo_model.py .\camera_detection.py .\analisis_estadistico.py
black .
isort .
ruff check .
```

Los comandos `black`, `isort` y `ruff` son opcionales y deben instalarse por separado si no estan disponibles.


Todo lo referente al dataset se encuentra en su repositorio oficial:
https://github.com/Whiffe/SCB-dataset

Se le agradece enormemente al autor por compartir su trabajo.

Dataset del proyecto: 

https://github.com/SalvadoRC4998/CUT_IA_Behavior


## Autor 
Leonel Isaias Sedano León 

Estudiante de la carrera de Ingeniería en Ciencias Commputacionales Universidad de Guadalajara Centro Universitario de Tonalá
