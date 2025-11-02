import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
from yolo_model import YOLOModel
import os
from analisis_estadistico import abrir_analisis_estadistico as open_statistics_window
from camera_detection import CameraDetector

class StudentDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎓 Detector de Comportamientos Estudiantiles")
        self.root.geometry("1200x800")
        self.root.configure(bg="#1a1a1a")
        self.root.minsize(800, 600)
        self.model = None
        self.yolo = YOLOModel(model_path="best.pt", device="cpu", conf=0.25)

        self.colors = {
            'bg_primary': '#1a1a1a',
            'bg_secondary': '#2d2d2d',
            'bg_tertiary': '#3d3d3d',
            'accent': '#00d4aa',
            'accent_hover': '#00b894',
            'warning': '#fdcb6e',
            'danger': '#e17055',
            'text_primary': '#ffffff',
            'text_secondary': '#b2b2b2',
            'success': '#00b894'
        }

        self.image_path = None
        self.processed_image = None
        self.detection_data = None

        self.setup_styles()
        self.create_interface()
        self.camera_detector = CameraDetector(self)
    
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Modern.TButton',
                       background=self.colors['accent'],
                       foreground=self.colors['text_primary'],
                       borderwidth=0,
                       focuscolor='none',
                       font=('Segoe UI', 10, 'bold'),
                       padding=(20, 10))
        style.map('Modern.TButton',
                 background=[('active', self.colors['accent_hover']),
                           ('pressed', self.colors['accent_hover'])])

    def create_interface(self):
        main_container = tk.Frame(self.root, bg=self.colors['bg_primary'])
        main_container.pack(fill=tk.BOTH, expand=True)

        header_frame = tk.Frame(main_container, bg=self.colors['bg_secondary'], height=80)
        header_frame.pack(fill=tk.X, padx=0, pady=0)
        header_frame.pack_propagate(False)

        title_frame = tk.Frame(header_frame, bg=self.colors['bg_secondary'])
        title_frame.pack(fill=tk.BOTH, expand=True)

        title_label = tk.Label(title_frame, 
                         text="🎓 Detector de Comportamientos Estudiantiles", 
                         font=("Segoe UI", 24, "bold"), 
                         bg=self.colors['bg_secondary'],
                         fg=self.colors['text_primary'])
        title_label.pack(pady=20)

        subtitle_label = tk.Label(title_frame,
                           text="Análisis inteligente de comportamientos en el aula",
                           font=("Segoe UI", 12),
                           bg=self.colors['bg_secondary'],
                           fg=self.colors['text_secondary'])
        subtitle_label.pack(pady=(0, 10))

        content_frame = tk.Frame(main_container, bg=self.colors['bg_primary'])
        content_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)

        control_panel = tk.Frame(content_frame, bg=self.colors['bg_secondary'], relief=tk.FLAT)
        control_panel.pack(fill=tk.X, pady=(0, 20))

        control_inner = tk.Frame(control_panel, bg=self.colors['bg_secondary'])
        control_inner.pack(fill=tk.X, padx=25, pady=20)

        control_title = tk.Label(control_inner,
                                text="Panel de Control",
                                font=("Segoe UI", 14, "bold"),
                                bg=self.colors['bg_secondary'],
                                fg=self.colors['text_primary'])
        control_title.pack(anchor=tk.W, pady=(0, 15))

        buttons_frame = tk.Frame(control_inner, bg=self.colors['bg_secondary'])
        buttons_frame.pack(fill=tk.X)

        self.btn_load = self.create_modern_button(
            buttons_frame, "📁 Cargar Imagen", self.load_image, self.colors['accent'])
        self.btn_load.pack(side=tk.LEFT, padx=(0, 15))

        self.btn_detect = self.create_modern_button(
            buttons_frame, "🔍 Detectar Comportamientos", self.detect_students, 
            self.colors['warning'], state='disabled')
        self.btn_detect.pack(side=tk.LEFT, padx=(0, 15))

        self.btn_save = self.create_modern_button(
            buttons_frame, "💾 Guardar Resultado", self.save_image, 
            self.colors['success'], state='disabled')
        self.btn_save.pack(side=tk.LEFT, padx=(0, 15))

        self.btn_stats = self.create_modern_button(
            buttons_frame, "📊 Análisis Estadístico", self.open_statistics, 
            "#9b59b6", state='disabled')
        self.btn_stats.pack(side=tk.LEFT)

        self.btn_camera = self.create_modern_button(
            buttons_frame, "📹 Detección Cámara", self._on_camera_button, 
            '#e74c3c')
        self.btn_camera.pack(side=tk.LEFT, padx=(15,0))

        image_container = tk.Frame(content_frame, bg=self.colors['bg_secondary'], relief=tk.FLAT)
        image_container.pack(fill=tk.BOTH, expand=True, pady=(0, 20))

        image_header = tk.Frame(image_container, bg=self.colors['bg_secondary'], height=50)
        image_header.pack(fill=tk.X)
        image_header.pack_propagate(False)

        image_title = tk.Label(image_header,
                              text="Vista Previa",
                              font=("Segoe UI", 14, "bold"),
                              bg=self.colors['bg_secondary'],
                              fg=self.colors['text_primary'])
        image_title.pack(pady=15, padx=25, anchor=tk.W)

        self.image_frame = tk.Frame(image_container, bg=self.colors['bg_tertiary'], relief=tk.FLAT)
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=25, pady=(0, 25))
        self.image_frame.pack_propagate(False)

        self.image_label = tk.Label(self.image_frame, 
                                    bg=self.colors['bg_tertiary'],
                                    fg=self.colors['text_secondary'],
                                    text="📷\n\nSelecciona una imagen para comenzar",
                                    font=("Segoe UI", 16),
                                    compound=tk.TOP)
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        results_panel = tk.Frame(content_frame, bg=self.colors['bg_secondary'], height=120)
        results_panel.pack(fill=tk.X)
        results_panel.pack_propagate(False)

        results_inner = tk.Frame(results_panel, bg=self.colors['bg_secondary'])
        results_inner.pack(fill=tk.BOTH, expand=True, padx=25, pady=20)

        results_title = tk.Label(results_inner,
                                text="📊 Resultados del Análisis",
                                font=("Segoe UI", 14, "bold"),
                                bg=self.colors['bg_secondary'],
                                fg=self.colors['text_primary'])
        results_title.pack(anchor=tk.W, pady=(0, 10))

        self.results_label = tk.Label(results_inner,
                                        text="Sin resultados disponibles",
                                        font=("Segoe UI", 12),
                                        bg=self.colors['bg_secondary'],
                                        fg=self.colors['text_secondary'])
        self.results_label.pack(anchor=tk.W)

        status_container = tk.Frame(main_container, bg=self.colors['bg_secondary'], height=40)
        status_container.pack(fill=tk.X, side=tk.BOTTOM)
        status_container.pack_propagate(False)

        self.status_bar = tk.Label(status_container,
                                    text="🟢 Listo para comenzar",
                                    font=("Segoe UI", 10),
                                    bg=self.colors['bg_secondary'],
                                    fg=self.colors['text_secondary'],
                                    anchor=tk.W)
        self.status_bar.pack(fill=tk.X, padx=25, pady=10)

    def create_modern_button(self, parent, text, command, color, state='normal'):
        btn = tk.Button(parent,
                       text=text,
                       command=command,
                       bg=color,
                       fg=self.colors['text_primary'],
                       font=("Segoe UI", 11, "bold"),
                       relief=tk.FLAT,
                       borderwidth=0,
                       padx=20,
                       pady=12,
                       cursor="hand2" if state == 'normal' else "arrow",
                       state=state)
        if state == 'normal':
            def on_enter(e):
                btn.config(bg=self.lighten_color(color))
            def on_leave(e):
                btn.config(bg=color)
            btn.bind("<Enter>", on_enter)
            btn.bind("<Leave>", on_leave)
        return btn

    def lighten_color(self, color):
        color_map = {
            self.colors['accent']: self.colors['accent_hover'],
            self.colors['warning']: '#e6ac00',
            self.colors['success']: '#009973'
        }
        return color_map.get(color, color)

    def update_button_state(self, button, enabled=True):
        if enabled:
            button.config(state='normal', cursor="hand2")
        else:
            button.config(state='disabled', cursor="arrow")

    def load_image(self):
        self.image_path = filedialog.askopenfilename(
            title="Seleccionar Imagen",
            filetypes=[("Archivos de imagen", "*.jpg *.jpeg *.png *.bmp")]
        )
        if self.image_path:
            self.show_image(self.image_path)
            self.update_button_state(self.btn_detect, True)
            self.status_bar.config(text=f"📁 Imagen cargada: {os.path.basename(self.image_path)}", 
                                   fg=self.colors['success'])

    def show_image(self, source, is_processed=False):
        imagen = Image.open(source) if not is_processed else Image.fromarray(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))
        self.image_frame.update_idletasks()
        frame_width = max(self.image_frame.winfo_width() - 40, 400)  
        frame_height = max(self.image_frame.winfo_height() - 40, 300)  
        ancho, alto = imagen.size
        scale_w = frame_width / ancho
        scale_h = frame_height / alto
        scale = min(scale_w, scale_h, 1.0)
        nuevo_ancho = int(ancho * scale)
        nuevo_alto = int(alto * scale)
        imagen_resized = imagen.resize((nuevo_ancho, nuevo_alto), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(imagen_resized)
        self.image_label.config(image=self.tk_image, text="")
        self.image_label.image = self.tk_image

    def detect_students(self):
        try:
            if not getattr(self, "image_path", None):
                messagebox.showwarning("Aviso", "Cargue una imagen antes de detectar")
                return
            image = cv2.imread(self.image_path)
            if image is None:
                messagebox.showerror("Error", "No se pudo leer la imagen")
                return
            try:
                res = self.yolo.detect_image(image)
            except FileNotFoundError as e:
                messagebox.showerror("Modelo no encontrado", f"No se encontró el modelo: {e}")
                return
            annotated = res.get("annotated")
            self.processed_image = annotated if annotated is not None else image
            boxes = res.get("boxes")
            count = int(boxes.shape[0]) if boxes is not None else 0
            self.detection_data = {
                "detections": boxes,
                "classes": res.get("names", {}),
                "total_count": count,
                "image_path": self.image_path,
            }
            self.show_image(self.processed_image, is_processed=True)
            self.update_button_state(self.btn_save, True)
            self.update_button_state(self.btn_stats, True)
            self.results_label.config(text=f"Detections: {self.detection_data['total_count']}")
            self.status_bar.config(text="✅ Detección completada", fg=self.colors['success'])
        except Exception as exc:
            messagebox.showerror("Error", f"Error durante la detección: {exc}")

    def open_statistics(self):
        try:
            open_statistics_window(self.root, self.detection_data)
        except Exception as e:
            messagebox.showerror("Error", f"Error al abrir análisis estadístico: {str(e)}")

    def save_image(self):
        if self.processed_image is None:
            messagebox.showerror("Error", "No hay imagen procesada para guardar")
            return
        save_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("Todos los archivos", "*.*")],
            title="Guardar imagen procesada"
        )
        if save_path:
            try:
                cv2.imwrite(save_path, self.processed_image)
                self.status_bar.config(text=f"💾 Imagen guardada: {os.path.basename(save_path)}", 
                                       fg=self.colors['success'])
                messagebox.showinfo("Exito", "Imagen guardada correctamente")
            except Exception as e:
                messagebox.showerror("Error", f"Error al guardar imagen: {str(e)}")

    def _on_camera_button(self):
        if self.camera_detector.running:
            self.camera_detector.stop()
            self.status_bar.config(text="⏹ Detención de cámara detenida", fg=self.colors['text_secondary'])
            self.btn_camera.config(text="📹 Detección Cámara")
        else:
            self.btn_camera.config(text="⏸ Detener Cámara")
            self.status_bar.config(text="🔄 Iniciando detección en cámara...", fg=self.colors['warning'])
            self.camera_detector.start()

if __name__ == "__main__":
    root = tk.Tk()
    app = StudentDetectionApp(root)
    root.mainloop()