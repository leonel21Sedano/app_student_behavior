import cv2
from ultralytics import YOLO

class CameraDetector:
   
    def __init__(self, parent, model_path='best.pt', device=0, conf=0.15):
        self.parent = parent
        self.model_path = model_path
        self.device = device
        self.conf = conf
        self.running = False
        self.thread = None

    def start(self):
        if self.running:
            return
        self.running = True
        # Start thread
        import threading
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        # Signal thread to stop
        self.running = False
        # The thread will clean up the camera and windows

    def _loop(self):
        try:
            # Lazy-load model if needed
            if getattr(self.parent, 'modelo', None) is None:
                try:
                    self.parent.barra_estado.config(text="⚙️ Cargando modelo de IA para cámara...")
                    self.parent.root.update()
                    self.parent.modelo = YOLO(self.model_path)
                except Exception as e:
                    print(f"Error loading model in CameraDetector: {e}")
                    self.running = False
                    self.parent.root.after(0, lambda: self.parent.barra_estado.config(text="❌ Error cargando modelo para cámara", fg=self.parent.colors.get('danger', '#e17055')))
                    return

            cap = cv2.VideoCapture(self.device, cv2.CAP_DSHOW)
            if not cap.isOpened():
                self.running = False
                self.parent.root.after(0, lambda: self.parent.barra_estado.config(text="❌ No se pudo abrir la cámara", fg=self.parent.colors.get('danger', '#e17055')))
                return

            self.parent.root.after(0, lambda: self.parent.barra_estado.config(text="👀 Detección en cámara activa. Presiona 'q' en la ventana para detener.", fg=self.parent.colors.get('success', '#00b894')))

            window_name = 'Deteccion en Tiempo Real'

            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                try:
                    results = self.parent.modelo(frame, conf=self.conf)
                    annotated = results[0].plot()
                except Exception as e:
                    print(f"Camera inference error: {e}")
                    annotated = frame

                cv2.imshow(window_name, annotated)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            try:
                cv2.destroyWindow(window_name)
            except Exception:
                pass

            self.running = False
            # Update UI from main thread
            self.parent.root.after(0, lambda: self.parent.barra_estado.config(text="⏹ Detección de cámara detenida", fg=self.parent.colors.get('text_secondary', '#b2b2b2')))
            self.parent.root.after(0, lambda: self.parent.btn_cam.config(text="📹 Detección Cámara"))

        except Exception as e:
            print(f"Error in camera loop: {e}")
            self.running = False
            self.parent.root.after(0, lambda: self.parent.barra_estado.config(text="❌ Error en detección de cámara", fg=self.parent.colors.get('danger', '#e17055')))
