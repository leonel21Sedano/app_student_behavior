# Detector de Comportamientos (YOLOv8) — Prototipo

Descripción
---------
Aplicación de escritorio (Tkinter) para detección con un modelo YOLO (ultralytics). Interfaz en español; lógica y variables en inglés. Incluye detección por imagen, detección en cámara y módulo de análisis estadístico con export a Excel.

Requisitos
---------
- Python 3.10+ recomendado
- Windows (instrucciones PowerShell)
- Dependencias listadas en `requirements.txt`
- Modelo YOLO: `best.pt` (no incluido en el repo)

Estructura principal
---------
- `app_deteccion_estudiantes.py` — GUI principal (labels y botones en español).
- `yolo_model.py` — wrapper del modelo (lazy load, device, conf).
- `camera_detection.py` — captura y loop de cámara (OpenCV, hilo).
- `analisis_estadistico.py` — ventana de análisis, gráficas y export a Excel.
- `requirements.txt` — dependencias.
- `.gitignore` — ya configurado (excluye modelos *.pt).

Uso (UI)
---------
- "📁 Cargar Imagen": abrir imagen.
- "🔍 Detectar Comportamientos": ejecutar inferencia sobre la imagen cargada.
- "💾 Guardar Resultado": guardar imagen anotada.
- "📊 Análisis Estadístico": abrir ventana con gráficas y opción "📄 Guardar en Excel".
- "📹 Detección Cámara": iniciar/detener detección en tiempo real (ventana OpenCV).

Notas importantes
---------
- No subir `best.pt` al repositorio. Usar `.gitignore` o Git LFS si lo necesitas.
- Si quieres usar GPU, instala la versión de `torch` adecuada y pasa `device="cuda"` al wrapper.
- La exportación a Excel usa `pandas` + `openpyxl` y crea hoja `Datos` + hoja `Grafico` con la imagen embebida.
- Normaliza colores (RGB/BGR) si integras la imagen en la GUI.

Buenas prácticas antes de subir
---------
- Ejecutar formateo y linter: `black`, `isort`, `ruff`.
- Ejecutar `pre-commit run --all-files` si instalaste hooks.
- Añadir `README.md`, `LICENSE` y `CONTRIBUTING.md` antes del primer push.


## Cómo funcionan las funciones principales

A continuación se explica de forma breve y directa qué hacen las funciones y clases más importantes del proyecto.

### app_deteccion_estudiantes.py — StudentDetectionApp
- __init__(self, root)  
  Inicializa la aplicación: configura la ventana, colores, widgets y crea instancias de `YOLOModel` y `CameraDetector`.

- load_image(self)  
  Abre un diálogo de archivo para seleccionar una imagen. Si se selecciona, guarda la ruta en `self.image_path`, muestra la imagen en la interfaz y habilita el botón de detección.

- show_image(self, source, is_processed=False)  
  Muestra una imagen en el panel principal. `source` puede ser una ruta (str) o un array BGR (cuando `is_processed=True`). Escala la imagen para que quepa en el marco y crea el objeto `ImageTk.PhotoImage` para Tkinter.

- detect_students(self)  
  Lee la imagen desde `self.image_path`, llama a `self.yolo.detect_image(image)` y procesa el resultado:
  - `annotated`: imagen anotada devuelta por el wrapper (si existe).
  - `boxes`: array con las detecciones.
  - Construye `self.detection_data` con keys: `detections`, `classes`, `total_count`, `image_path`.
  - Actualiza la vista previa y habilita botones de guardado/análisis.

- save_image(self)  
  Abre diálogo para guardar la imagen procesada (`self.processed_image`) en disco usando `cv2.imwrite`.

- open_statistics(self)  
  Llama a `abrir_analisis_estadistico(self.root, self.detection_data)` para abrir la ventana de análisis estadístico.

- _on_camera_button(self)  
  Inicia o detiene la detección por cámara delegando en `CameraDetector`. Actualiza texto del botón y barra de estado.

- create_modern_button / update_button_state / lighten_color  
  Helpers para crear botones con estilo y controlar su estado (habilitado/deshabilitado) y apariencia.

---

### yolo_model.py — YOLOModel
- __init__(self, model_path="best.pt", device="cpu", conf=0.25)  
  Guarda parámetros (ruta modelo, dispositivo y umbral de confianza). No carga el modelo aún.

- _ensure_model(self)  
  Carga el modelo la primera vez que haga falta. Valida que `model_path` exista y crea la instancia `YOLO(...)`. Intenta mover el modelo al `device` indicado si procede.

- detect_image(self, image_bgr)  
  Ejecuta inferencia sobre una imagen BGR (numpy array) usando el modelo cargado:
  - Llama a `_ensure_model()`.
  - Ejecuta `results = self._model(image_bgr, conf=self.conf)` y toma `results[0]`.
  - Intenta obtener `annotated` con `r0.plot()`.
  - Extrae `boxes` con `r0.boxes.data.cpu().numpy()` si está disponible.
  - Devuelve dict: `{"annotated", "boxes", "names", "raw"}`.

- set_confidence(self, conf) / load(self) / unload(self)  
  Cambian el umbral, fuerzan carga o descargan el modelo de memoria respectivamente.

Notas: `detect_image` no asume formato RGB/BGR para `annotated`; el llamador normaliza antes de mostrar.

---

### camera_detection.py — CameraDetector
- __init__(self, parent, model_path='best.pt', device=0, conf=0.15)  
  Guarda referencias a la app (parent), parámetros del modelo y estado (running/thread).

- start(self)  
  Marca `running = True` y lanza un hilo daemon que ejecuta `_loop()`.

- stop(self)  
  Marca `running = False`; el hilo detectará el flag y cerrará captura/ventanas.

- _loop(self)  
  Bucle de captura:
  - Carga el modelo si es necesario (lazy load).
  - Abre `cv2.VideoCapture(self.device, cv2.CAP_DSHOW)` y lee frames en bucle.
  - Para cada frame llama al modelo (`self.parent.modelo(frame, conf=self.conf)`), obtiene la imagen anotada y la muestra con `cv2.imshow`.
  - Escucha la tecla 'q' para detener; al terminar libera la cámara y cierra la ventana.
  - Actualiza la UI (barra de estado, botones) usando `self.parent.root.after(...)` para ejecutar en el hilo principal.

---

### analisis_estadistico.py — StatisticalAnalysisWindow
- __init__(self, parent, detection_data=None)  
  Crea la ventana de análisis y guarda `detection_data`.

- generate_sample_data(self)  
  Si `detection_data` contiene detecciones válidas: convierte las detecciones a conteos por clase (usa la columna de clase en `boxes[:,5]`) y devuelve un dict `{label: count}`. Si no hay datos válidos devuelve datos de ejemplo.

- generate_analysis(self)  
  Construye y muestra en la ventana:
  - Un gráfico de barras con frecuencias por comportamiento.
  - Un gráfico de pastel con distribución porcentual.
  - Un panel resumen con estadísticas (total, promedio, más/menos frecuente).
  - Inserta la figura en la GUI con `FigureCanvasTkAgg`.

- create_stats_panel(self, data)  
  Muestra texto resumen con total, promedio, ítem más y menos frecuente.

- export_graphs(self)  
  Permite guardar los gráficos como PNG/PDF o como Excel (.xlsx). Para Excel:
  - Crea un DataFrame con `pandas`.
  - Guarda hoja `Datos`.
  - Renderiza la figura a PNG en memoria y la inserta en hoja `Grafico` usando `openpyxl`.

- export_to_excel(self)  
  Similar a `export_graphs` cuando se elige `.xlsx`: crea archivo Excel con hoja de datos y hoja con la imagen del gráfico embebida.

---

### Formato de `detection_data`
La app usa el siguiente dict estándar para pasar resultados entre módulos:

- `detections` : numpy.ndarray o None (cada fila suele ser [x1, y1, x2, y2, conf, cls])  
- `classes` : mapping id -> label (dict)  
- `total_count` : int  
- `image_path` : str

Usar siempre comprobaciones explícitas antes de tratar arrays (ej. `if boxes is not None and boxes.size > 0:`).






