from ultralytics import YOLO # type: ignore

class YoloDetector:
    def __init__(self, model_path: str):
        
        self.model = YOLO(model_path)
        self.features = []
        self.target_layer = self.model.model.model[10] # type: ignore
        self.target_layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self.features.append(output)

    def detect(self, frame):
        self.features.clear()
        results = self.model(frame,classes=[0], verbose=False)
        if not self.features:
            print("[WARN] Aucune feature capturée — vérifie le hook.")
            return results, None
        return results, self.features[-1]  # retourne la feature map
