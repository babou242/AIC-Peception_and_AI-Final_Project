import torch
import cv2
import numpy as np
import torchreid as te

class ReIDExtractor:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.extractor = te.utils.FeatureExtractor(
            model_name='osnet_x0_25',
            device=device
        )
        self.device = device

    def get_embeddings(self, frame, boxes):
        """
        Extrait les embeddings (512D) pour chaque personne détectée par YOLO.
        """
        crops = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crops.append(crop)

        if not crops:
            return []

        embeddings = self.extractor(crops)
        embeddings = embeddings.cpu().numpy()
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings
