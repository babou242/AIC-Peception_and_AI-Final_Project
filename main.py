import cv2
import time
from src.detector import YoloDetector
from src.reid_extractor import ReIDExtractor
from src.database import EmbeddingDB
from src.utils import draw_boxes
import config

def main():
    print("🔄 Initialisation...")
    print(f"📋 Configuration:")
    print(f"  - Modèle ReID: {config.REID_MODEL}")
    print(f"  - Seuil similarité: {config.SIMILARITY_THRESHOLD}")
    print(f"  - Timeout: {config.ID_TIMEOUT}s")
    
    # Utiliser YOLO pour la détection et ReID (OSNet) pour l'identification
    detector = YoloDetector(config.YOLO_MODEL)
    reid_extractor = ReIDExtractor(model_name=config.REID_MODEL)
    db = EmbeddingDB(
        threshold=config.SIMILARITY_THRESHOLD,
        max_embeddings=config.MAX_EMBEDDINGS_PER_PERSON,
        timeout=config.ID_TIMEOUT
    )
    
    print("✅ Modèles chargés (YOLO + OSNet)")

    cap = cv2.VideoCapture(config.VIDEO_SOURCE)
    if not cap.isOpened():
        print("❌ Erreur caméra.")
        return
    
    # Réduire la résolution pour de meilleures performances
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    
    prev_time = time.time()
    fps = 0
    
    print("🎥 Démarrage du tracking ReID... (Échap pour quitter)")

    while True:
        ret, frame = cap.read()
        if not ret: 
            break

        # 1. Détection avec YOLO
        results, _ = detector.detect(frame)
        boxes = results[0].boxes.xyxy
        
        if len(boxes) == 0:
            # Pas de détection
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Tracking ReID (YOLO + OSNet)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue
        
        # 2. Extraction d'embeddings avec OSNet (meilleur que YOLO features)
        embeddings = reid_extractor.get_embeddings(frame, boxes)
        
        if len(embeddings) == 0:
            # Pas d'embeddings valides
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Tracking ReID (YOLO + OSNet)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue
        
        # 3. Matching et assignation d'IDs avec mise à jour
        ids = []
        confidences = []
        
        for emb in embeddings:
            # Chercher un match
            match_id = db.find_match(emb, min_confidence=config.MIN_CONFIDENCE)
            
            if match_id is None:
                # Nouvelle personne
                match_id = db.add(emb)
                confidence = 1.0  # Nouveau = haute confiance
            else:
                # Personne existante : mettre à jour son embedding
                db.update(match_id, emb)
                confidence = db.get_confidence(match_id, emb)
            
            ids.append(match_id)
            confidences.append(confidence)
        
        # 4. Affichage avec confiance
        frame = draw_boxes(frame, boxes, ids, confidences)
        
        # Calcul FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        # Afficher les statistiques
        stats = db.get_stats()
        info_text = [
            f"FPS: {fps:.1f}",
            f"Personnes actuelles: {len(ids)}",
            f"IDs actifs: {stats['total_persons']}",
            f"Nouveaux IDs: {stats['total_new_ids']}",
            f"Matches: {stats['total_matches']}"
        ]
        
        y_offset = 30
        for i, text in enumerate(info_text):
            cv2.putText(frame, text, (10, y_offset + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow("Tracking ReID (YOLO + OSNet)", frame)
        
        # Touche 'r' pour reset la base
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('r'):  # Reset
            db.reset()
            print("🔄 Base de données réinitialisée")

    cap.release()
    cv2.destroyAllWindows()
    
    # Afficher les stats finales
    final_stats = db.get_stats()
    print("\n📊 Statistiques finales:")
    print(f"  - Total personnes uniques: {final_stats['total_new_ids']}")
    print(f"  - Total matches: {final_stats['total_matches']}")
    print(f"  - Total mises à jour: {final_stats['total_updates']}")
    print("✅ Fin du traitement.")

if __name__ == "__main__":
    main()
