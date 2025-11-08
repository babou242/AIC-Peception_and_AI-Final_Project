import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingDB:
    def __init__(self, threshold=0.8):
        self.embeddings = []
        self.ids = []
        self.next_id = 0
        self.threshold = threshold

    def find_match(self, embedding):
        if not self.embeddings:
            return None
        sims = cosine_similarity([embedding], self.embeddings) # type: ignore
        best_idx = np.argmax(sims)
        if sims[0, best_idx] > self.threshold:
            return self.ids[best_idx]
        return None

    def add(self, embedding):
        person_id = self.next_id
        self.next_id += 1
        self.embeddings.append(embedding)
        self.ids.append(person_id)
        return person_id
