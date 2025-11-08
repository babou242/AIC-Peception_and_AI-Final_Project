import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import time

class EmbeddingDB:
    """
    Base de données d'embeddings avec gestion avancée :
    - Moyenne glissante (multiple embeddings par personne)
    - Score de confiance
    - Timeout pour IDs inactifs
    """
    
    def __init__(self, 
                 threshold=0.65,           # Seuil de similarité (abaissé)
                 max_embeddings=10,        # Nombre max d'embeddings par personne
                 update_rate=0.3,          # Taux de mise à jour (0-1)
                 timeout=30.0):            # Timeout en secondes
        
        self.threshold = threshold
        self.max_embeddings = max_embeddings
        self.update_rate = update_rate
        self.timeout = timeout
        
        # Stockage : {person_id: [list of embeddings]}
        self.embeddings_history = defaultdict(list)
        
        # Timestamps : {person_id: last_seen_time}
        self.last_seen = {}
        
        # Compteur d'IDs
        self.next_id = 0
        
        # Stats pour debug
        self.stats = {
            'total_matches': 0,
            'total_new_ids': 0,
            'total_updates': 0
        }

    def find_match(self, embedding, min_confidence=0.6):
        """
        Trouve le meilleur match pour un embedding donné.
        
        Args:
            embedding: Embedding à matcher
            min_confidence: Confiance minimale requise
            
        Returns:
            person_id ou None
        """
        if not self.embeddings_history:
            return None
        
        # Nettoyer les IDs expirés
        self._cleanup_expired_ids()
        
        # Calculer la similarité avec la moyenne de chaque personne
        best_id = None
        best_similarity = -1
        
        for person_id, emb_list in self.embeddings_history.items():
            # Calculer la moyenne des embeddings de cette personne
            mean_embedding = np.mean(emb_list, axis=0)
            
            # Similarité cosinus
            similarity = np.dot(embedding, mean_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_id = person_id
        
        # Vérifier le seuil et la confiance
        if best_similarity > self.threshold and best_similarity > min_confidence:
            self.stats['total_matches'] += 1
            return best_id
        
        return None

    def add(self, embedding):
        """
        Ajoute une nouvelle personne à la base de données.
        
        Args:
            embedding: Embedding de la nouvelle personne
            
        Returns:
            person_id
        """
        person_id = self.next_id
        self.next_id += 1
        
        self.embeddings_history[person_id].append(embedding)
        self.last_seen[person_id] = time.time()
        
        self.stats['total_new_ids'] += 1
        
        return person_id

    def update(self, person_id, new_embedding):
        """
        Met à jour l'embedding d'une personne existante.
        Utilise une moyenne glissante pour stabiliser.
        
        Args:
            person_id: ID de la personne
            new_embedding: Nouvel embedding à ajouter
        """
        if person_id not in self.embeddings_history:
            return
        
        # Ajouter le nouvel embedding
        self.embeddings_history[person_id].append(new_embedding)
        
        # Limiter le nombre d'embeddings stockés
        if len(self.embeddings_history[person_id]) > self.max_embeddings:
            # Garder les plus récents
            self.embeddings_history[person_id] = \
                self.embeddings_history[person_id][-self.max_embeddings:]
        
        # Mettre à jour le timestamp
        self.last_seen[person_id] = time.time()
        
        self.stats['total_updates'] += 1

    def _cleanup_expired_ids(self):
        """Supprime les IDs qui n'ont pas été vus depuis longtemps"""
        current_time = time.time()
        expired_ids = []
        
        for person_id, last_time in self.last_seen.items():
            if current_time - last_time > self.timeout:
                expired_ids.append(person_id)
        
        for person_id in expired_ids:
            del self.embeddings_history[person_id]
            del self.last_seen[person_id]
    
    def get_mean_embedding(self, person_id):
        """Retourne l'embedding moyen d'une personne"""
        if person_id not in self.embeddings_history:
            return None
        return np.mean(self.embeddings_history[person_id], axis=0)
    
    def get_confidence(self, person_id, current_embedding):
        """
        Calcule la confiance de l'identification.
        Plus le score est élevé, plus on est sûr de l'identification.
        """
        mean_emb = self.get_mean_embedding(person_id)
        if mean_emb is None:
            return 0.0
        
        similarity = np.dot(current_embedding, mean_emb)
        
        # Bonus basé sur le nombre d'observations
        num_observations = len(self.embeddings_history[person_id])
        observation_bonus = min(num_observations / 10.0, 0.1)
        
        return similarity + observation_bonus
    
    def get_stats(self):
        """Retourne les statistiques de la base"""
        return {
            **self.stats,
            'total_persons': len(self.embeddings_history),
            'active_ids': list(self.embeddings_history.keys())
        }
    
    def reset(self):
        """Réinitialise la base de données"""
        self.embeddings_history.clear()
        self.last_seen.clear()
        self.next_id = 0
        self.stats = {
            'total_matches': 0,
            'total_new_ids': 0,
            'total_updates': 0
        }
