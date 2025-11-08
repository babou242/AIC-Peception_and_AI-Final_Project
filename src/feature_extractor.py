import torch
import numpy as np

class FeatureExtractor:
    def __init__(self):
        pass

    def get_embeddings(self, frame, feature_map, boxes):
        h_img, w_img = frame.shape[:2]
        _, c, h_feat, w_feat = feature_map.shape
        x_scale, y_scale = w_feat / w_img, h_feat / h_img

        embeddings = []
        for box in boxes:
            x1, y1, x2, y2 = box.cpu().numpy()
            fx1, fy1, fx2, fy2 = map(int, [x1*x_scale, y1*y_scale, x2*x_scale, y2*y_scale])
            region = feature_map[0, :, fy1:fy2, fx1:fx2]
            if region.numel() == 0:
                continue
            emb = torch.mean(region, dim=(1, 2))
            emb = emb / torch.norm(emb)  # normalisation L2
            embeddings.append(emb.cpu().numpy())
        return np.array(embeddings)
