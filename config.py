"""
Configuration pour le système de tracking ReID

Ajustez ces paramètres selon votre environnement et vos besoins.
"""

# ==================== Modèles ====================

# Modèle YOLO
YOLO_MODEL = "models/yolov8n.pt"  # yolov8n.pt (rapide) ou yolov8s.pt (précis)

# Modèle ReID OSNet
REID_MODEL = "osnet_x0_5"  # Options: osnet_x0_25 (rapide), osnet_x0_5, osnet_x0_75, osnet_x1_0 (précis)

# ==================== Paramètres de Matching ====================

# Seuil de similarité pour le matching (0-1)
# Plus élevé = moins de faux positifs, mais plus de faux négatifs
# Plus bas = moins de faux négatifs, mais plus de faux positifs
SIMILARITY_THRESHOLD = 0.65  # Recommandé: 0.60-0.70

# Confiance minimale pour accepter un match
MIN_CONFIDENCE = 0.60

# Nombre maximum d'embeddings par personne (moyenne glissante)
MAX_EMBEDDINGS_PER_PERSON = 10  # Recommandé: 5-15

# Timeout avant d'oublier un ID inactif (secondes)
ID_TIMEOUT = 30.0  # Recommandé: 20-60

# ==================== Caméra ====================

# Résolution de la caméra
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Source vidéo (0 = webcam par défaut, ou chemin vers fichier vidéo)
VIDEO_SOURCE = 0

# ==================== Affichage ====================

# Afficher les statistiques à l'écran
SHOW_STATS = True

# Afficher la confiance sur les bounding boxes
SHOW_CONFIDENCE = True

# Couleurs des bounding boxes selon la confiance
COLOR_HIGH_CONFIDENCE = (0, 255, 0)      # Vert
COLOR_MEDIUM_CONFIDENCE = (0, 255, 255)  # Jaune
COLOR_LOW_CONFIDENCE = (0, 165, 255)     # Orange

# ==================== Performance ====================

# Activer CUDA si disponible
USE_CUDA = True

# Batch size pour l'extraction ReID (si plusieurs personnes)
REID_BATCH_SIZE = 8

# ==================== Réglages avancés ====================

# Taille minimale des crops (pixels)
MIN_CROP_WIDTH = 30
MIN_CROP_HEIGHT = 60

# Filtrage des détections YOLO
YOLO_CONFIDENCE = 0.5  # Confiance minimale YOLO
YOLO_IOU_THRESHOLD = 0.5  # NMS IoU threshold

# ==================== Recommandations par environnement ====================

"""
ENVIRONNEMENT CONTRÔLÉ (Bureau, Magasin):
- SIMILARITY_THRESHOLD = 0.70
- MIN_CONFIDENCE = 0.65
- ID_TIMEOUT = 60.0
- REID_MODEL = "osnet_x0_5"

ENVIRONNEMENT VARIABLE (Rue, Foule):
- SIMILARITY_THRESHOLD = 0.60
- MIN_CONFIDENCE = 0.55
- ID_TIMEOUT = 20.0
- REID_MODEL = "osnet_x0_75" ou "osnet_x1_0"

PERFORMANCE MAXIMALE (CPU faible):
- REID_MODEL = "osnet_x0_25"
- CAMERA_WIDTH = 480
- CAMERA_HEIGHT = 360
- MAX_EMBEDDINGS_PER_PERSON = 5

PRÉCISION MAXIMALE (GPU disponible):
- REID_MODEL = "osnet_x1_0"
- SIMILARITY_THRESHOLD = 0.65
- MAX_EMBEDDINGS_PER_PERSON = 15
- USE_CUDA = True
"""
