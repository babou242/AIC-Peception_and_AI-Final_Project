import torch
import cv2
import numpy as np

try:
    import torchreid
except ImportError:
    print("⚠️  torchreid non installé. Installez avec: uv pip install torchreid")
    raise

class ReIDExtractor:
    """
    Extracteur d'embeddings ReID utilisant OSNet.
    
    OSNet (Omni-Scale Network) est spécialement conçu pour le ReID et offre
    de bien meilleures performances que les feature maps de YOLO.
    
    Args:
        model_name: 'osnet_x0_25' (léger), 'osnet_x0_5', 'osnet_x0_75', 'osnet_x1_0'
        device: 'cuda' ou 'cpu'
    """
    
    def __init__(self, model_name='osnet_x0_5', device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_name = model_name
        
        print(f"🔄 Chargement du modèle ReID: {model_name} sur {device}")
        
        try:
            self.extractor = torchreid.utils.FeatureExtractor(
                model_name=model_name,
                device=device,
                verbose=False
            )
            print(f"✅ Modèle ReID chargé avec succès")
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle ReID: {e}")
            raise

    def get_embeddings(self, frame, boxes):
        """
        Extrait les embeddings ReID pour chaque personne détectée.
        
        Args:
            frame: Image BGR (format OpenCV)
            boxes: Tensor de bounding boxes [N, 4] au format xyxy
            
        Returns:
            np.array: Embeddings normalisés [N, 512]
        """
        if len(boxes) == 0:
            return np.array([])
        
        crops = []
        valid_indices = []
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            
            # Vérifier que les coordonnées sont valides
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Extraire le crop
            crop = frame[y1:y2, x1:x2]
            
            # Vérifier que le crop est valide (taille minimale)
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                print(f"⚠️  Crop {i} trop petit ou vide: {crop.shape}")
                continue
            
            # Convertir BGR → RGB (important pour torchreid)
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crops.append(crop_rgb)
            valid_indices.append(i)

        if not crops:
            return np.array([])

        try:
            # Extraction des embeddings avec OSNet
            embeddings = self.extractor(crops)
            
            # Convertir en numpy
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()
            
            # Normalisation L2 (importante pour la similarité cosinus)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Éviter division par zéro
            embeddings = embeddings / norms
            
            return embeddings
            
        except Exception as e:
            print(f"❌ Erreur lors de l'extraction des embeddings: {e}")
            import traceback
            traceback.print_exc()
            return np.array([])
    
    def compute_similarity(self, emb1, emb2):
        """
        Calcule la similarité cosinus entre deux embeddings.
        
        Args:
            emb1, emb2: Embeddings normalisés
            
        Returns:
            float: Similarité cosinus [0, 1]
        """
        return np.dot(emb1, emb2)
    
    def get_model_info(self):
        """Retourne les informations sur le modèle"""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'embedding_dim': 512,  # OSNet produit des embeddings 512D
            'input_size': (256, 128)  # H x W
        }
