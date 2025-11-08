import cv2
from src.detector import YoloDetector
from src.feature_extractor import FeatureExtractor
from src.database import EmbeddingDB
from src.utils import draw_boxes

def main():
    detector = YoloDetector("models/yolov8n.pt")
    extractor = FeatureExtractor()
    db = EmbeddingDB()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur caméra.")
        return

    while True:
        ret, frame = cap.read()
        if not ret: break

        results, fmap = detector.detect(frame)
        boxes = results[0].boxes.xyxy
        embeddings = extractor.get_embeddings(frame, fmap, boxes)

        ids = []
        for emb in embeddings:
            match_id = db.find_match(emb)
            if match_id is None:
                match_id = db.add(emb)
            ids.append(match_id)

        frame = draw_boxes(frame, boxes, ids)
        cv2.imshow("Tracking ReID", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
