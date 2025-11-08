"""
Exemple d'utilisation du système YOLO + ReID (OSNet)

Ce script démontre comment combiner YOLOv8 pour la détection 
et OSNet pour le ReID afin d'obtenir un tracking robuste.
"""

import cv2
import numpy as np
from src.detector import YoloDetector
from src.reid_extractor import ReIDExtractor

def test_reid_extraction():
    """Test basique de l'extraction ReID avec OSNet"""
    
    print("🔄 Test d'extraction ReID...")
    
    # Initialisation
    detector = YoloDetector("models/yolov8n.pt")
    reid = ReIDExtractor()
    
    # Test avec une image
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Impossible de capturer une frame")
        return
    
    # Détection
    results, _ = detector.detect(frame)
    boxes = results[0].boxes.xyxy
    
    print(f"📦 {len(boxes)} personne(s) détectée(s)")
    
    if len(boxes) == 0:
        print("⚠️  Aucune personne détectée, placez-vous devant la caméra")
        return
    
    # Extraction ReID
    try:
        embeddings = reid.get_embeddings(frame, boxes)
        print(f"✅ Embeddings extraits : {len(embeddings)}")
        
        for i, emb in enumerate(embeddings):
            print(f"  Person {i+1}:")
            print(f"    - Shape: {emb.shape}")
            print(f"    - Norm: {np.linalg.norm(emb):.4f}")  # Devrait être ≈1.0
            print(f"    - Min/Max: [{emb.min():.4f}, {emb.max():.4f}]")
            
            # Vérification que l'embedding est normalisé
            if abs(np.linalg.norm(emb) - 1.0) > 0.01:
                print(f"    ⚠️  WARNING: Embedding pas bien normalisé!")
            else:
                print(f"    ✅ Embedding normalisé correctement")
                
    except Exception as e:
        print(f"❌ Erreur lors de l'extraction : {e}")
        import traceback
        traceback.print_exc()

def test_similarity():
    """Test de similarité entre embeddings de la même personne"""
    
    print("\n🔄 Test de similarité...")
    
    detector = YoloDetector("models/yolov8n.pt")
    reid = ReIDExtractor()
    
    cap = cv2.VideoCapture(0)
    
    embeddings_list = []
    
    print("📸 Capture de 5 frames pour test de similarité...")
    for i in range(5):
        ret, frame = cap.read()
        if not ret:
            continue
            
        results, _ = detector.detect(frame)
        boxes = results[0].boxes.xyxy
        
        if len(boxes) > 0:
            embeddings = reid.get_embeddings(frame, boxes)
            if len(embeddings) > 0:
                embeddings_list.append(embeddings[0])  # Première personne
                print(f"  Frame {i+1}: Embedding capturé ✓")
    
    cap.release()
    
    if len(embeddings_list) < 2:
        print("⚠️  Pas assez d'embeddings capturés")
        return
    
    # Calcul de similarité cosinus
    from sklearn.metrics.pairwise import cosine_similarity
    
    print("\n📊 Matrice de similarité :")
    for i in range(len(embeddings_list)):
        for j in range(i+1, len(embeddings_list)):
            sim = cosine_similarity([embeddings_list[i]], [embeddings_list[j]])[0, 0]
            print(f"  Frame {i+1} vs Frame {j+1}: {sim:.4f}")
            
            if sim > 0.7:
                print(f"    ✅ Bonne similarité (même personne)")
            else:
                print(f"    ⚠️  Faible similarité (possible faux négatif)")

def visualize_crops():
    """Visualise les crops envoyés au modèle ReID"""
    
    print("\n🔄 Visualisation des crops...")
    
    detector = YoloDetector("models/yolov8n.pt")
    
    cap = cv2.VideoCapture(0)
    
    print("📸 Appuyez sur ESPACE pour capturer, ESC pour quitter")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results, _ = detector.detect(frame)
        boxes = results[0].boxes.xyxy
        
        # Dessiner les boxes
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Person {i+1}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow("Detection", frame)
        
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            # Afficher les crops
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.cpu().numpy())
                crop = frame[y1:y2, x1:x2]
                
                if crop.size > 0:
                    # Redimensionner pour affichage
                    crop_resized = cv2.resize(crop, (128, 256))
                    cv2.imshow(f"Person {i+1} Crop", crop_resized)
                    
                    print(f"Crop {i+1}: {crop.shape}")
    
    cap.release()
    cv2.destroyAllWindows()

def compare_yolo_vs_reid():
    """Compare la qualité des embeddings YOLO vs ReID"""
    
    print("\n🔄 Comparaison YOLO Features vs OSNet ReID...")
    
    detector = YoloDetector("models/yolov8n.pt")
    reid = ReIDExtractor()
    
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return
    
    results, feature_map = detector.detect(frame)
    boxes = results[0].boxes.xyxy
    
    if len(boxes) == 0:
        print("⚠️  Aucune détection")
        return
    
    # YOLO Features
    print("\n📊 YOLO Features:")
    if feature_map is not None:
        print(f"  Shape: {feature_map.shape}")
        print(f"  Channels: {feature_map.shape[1]}")
        print(f"  Spatial resolution: {feature_map.shape[2]}x{feature_map.shape[3]}")
        
        # Extraction basique
        box = boxes[0]
        x1, y1, x2, y2 = box.cpu().numpy()
        h_img, w_img = frame.shape[:2]
        _, c, h_feat, w_feat = feature_map.shape
        x_scale, y_scale = w_feat / w_img, h_feat / h_img
        
        fx1, fy1 = int(x1*x_scale), int(y1*y_scale)
        fx2, fy2 = int(x2*x_scale), int(y2*y_scale)
        
        import torch
        region = feature_map[0, :, fy1:fy2, fx1:fx2]
        yolo_emb = torch.mean(region, dim=(1, 2)).cpu().numpy()
        yolo_emb = yolo_emb / np.linalg.norm(yolo_emb)
        
        print(f"  Embedding dim: {yolo_emb.shape}")
        print(f"  Values range: [{yolo_emb.min():.4f}, {yolo_emb.max():.4f}]")
    
    # OSNet ReID
    print("\n📊 OSNet ReID:")
    embeddings = reid.get_embeddings(frame, boxes)
    reid_emb = embeddings[0]
    
    print(f"  Embedding dim: {reid_emb.shape}")
    print(f"  Values range: [{reid_emb.min():.4f}, {reid_emb.max():.4f}]")
    print(f"  Norm: {np.linalg.norm(reid_emb):.4f}")
    
    print("\n💡 Conclusion:")
    print("  - YOLO: Features spatiales, bonne pour localisation")
    print("  - ReID: Features discriminantes, excellentes pour identification")

if __name__ == "__main__":
    print("=" * 60)
    print("Test du système YOLO + ReID")
    print("=" * 60)
    
    # Test 1: Extraction basique
    test_reid_extraction()
    
    # Test 2: Similarité
    test_similarity()
    
    # Test 3: Visualisation (interactif)
    # visualize_crops()
    
    # Test 4: Comparaison
    compare_yolo_vs_reid()
    
    print("\n" + "=" * 60)
    print("✅ Tests terminés")
    print("=" * 60)
