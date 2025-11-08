import cv2
import numpy as np

def draw_boxes(frame, boxes, ids, confidences=None):
    """
    Dessine les bounding boxes avec IDs et confiance.
    
    Args:
        frame: Image à annoter
        boxes: Bounding boxes [N, 4]
        ids: Liste des IDs
        confidences: Liste des scores de confiance (optionnel)
    """
    if confidences is None:
        confidences = [1.0] * len(ids)
    
    for box, pid, conf in zip(boxes, ids, confidences):
        x1, y1, x2, y2 = map(int, box)
        
        # Couleur basée sur la confiance
        if conf > 0.8:
            color = (0, 255, 0)    # Vert - Haute confiance
        elif conf > 0.65:
            color = (0, 255, 255)  # Jaune - Confiance moyenne
        else:
            color = (0, 165, 255)  # Orange - Faible confiance
        
        # Dessiner le rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Label avec ID et confiance
        label = f"ID {pid} ({conf:.2f})"
        
        # Fond pour le texte
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1-text_h-10), (x1+text_w+10, y1), color, -1)
        
        # Texte
        cv2.putText(frame, label, (x1+5, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return frame
