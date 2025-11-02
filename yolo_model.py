from typing import Any, Dict, Optional
import os
import logging
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class YOLOModel:
    def __init__(self, model_path: str = "best.pt", device: str = "cpu", conf: float = 0.25):
        self.model_path = model_path
        self.device = device
        self.conf = float(conf)
        self._model: Optional[YOLO] = None

    def _ensure_model(self) -> None:
        if self._model is None:
            if not os.path.exists(self.model_path):
                logger.error("Modelo no encontrado %s", self.model_path)
                raise FileNotFoundError(self.model_path)
            self._model = YOLO(self.model_path)
            try:
                if self.device and self.device != "cpu":
                    self._model.to(self.device)
            except Exception:
                pass

    def detect_image(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        self._ensure_model()
        results = self._model(image_bgr, conf=self.conf)
        r0 = results[0]
        annotated = None
        try:
            annotated = r0.plot()
        except Exception:
            annotated = None
        boxes = None
        try:
            if hasattr(r0, "boxes") and r0.boxes is not None:
                boxes = r0.boxes.data.cpu().numpy()
        except Exception:
            boxes = None
        names = getattr(r0, "names", {}) or {}
        return {"annotated": annotated, "boxes": boxes, "names": names, "raw": r0}

    def set_confidence(self, conf: float) -> None:
        self.conf = float(conf)

    def load(self) -> None:
        self._ensure_model()

    def unload(self) -> None:
        try:
            del self._model
        finally:
            self._model = None