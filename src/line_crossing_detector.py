import cv2
import numpy as np
from collections import defaultdict


class LineCrossingDetector:
    """
    Détecte quand une personne franchit une ligne virtuelle définie.
    """
    
    def __init__(self, line_start, line_end, frame_width, frame_height):
        """
        Initialise le détecteur de franchissement de ligne.
        
        Args:
            line_start: Tuple (x, y) du point de départ de la ligne
            line_end: Tuple (x, y) du point de fin de la ligne
            frame_width: Largeur du frame
            frame_height: Hauteur du frame
        """
        self.line_start = np.array(line_start, dtype=np.float32)
        self.line_end = np.array(line_end, dtype=np.float32)
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Historique des positions pour chaque ID
        self.person_history = defaultdict(list)
        
        # Personnes qui ont traversé la ligne
        self.crossed_persons = set()
        self.crossing_events = []  # [(id, timestamp, direction)]
        
    def get_point_to_line_distance(self, point, line_start, line_end):
        """
        Calcule la distance signée d'un point à une ligne.
        La distance est positive d'un côté et négative de l'autre.
        
        Pour une ligne verticale ou horizontale, cela donne la distance algébrique.
        """
        line_vec = line_end - line_start
        point_vec = point - line_start
        
        # Normaliser le vecteur de la ligne
        line_len_sq = np.dot(line_vec, line_vec)
        
        if line_len_sq == 0:
            # Les deux points de la ligne sont identiques
            return np.linalg.norm(point_vec)
        
        # Produit vectoriel 2D (retourne un scalaire en 2D)
        # cross(a, b) = a.x * b.y - a.y * b.x
        cross = line_vec[0] * point_vec[1] - line_vec[1] * point_vec[0]
        
        # Normaliser par la longueur de la ligne pour obtenir une distance
        distance = cross / np.sqrt(line_len_sq)
        
        return distance
    
    def check_crossing(self, person_id, center_x, center_y):
        """
        Vérifie si une personne a franchi la ligne.
        
        Args:
            person_id: ID de la personne
            center_x: Coordonnée X du centre de la bounding box
            center_y: Coordonnée Y du centre de la bounding box
            
        Returns:
            Tuple (has_crossed, direction) où direction est "→" (droite) ou "←" (gauche)
        """
        current_pos = np.array([center_x, center_y], dtype=np.float32)
        
        # Calculer la distance du point à la ligne
        distance = self.get_point_to_line_distance(current_pos, self.line_start, self.line_end)
        
        has_crossed = False
        direction = None
        
        # Vérifier l'historique
        if person_id in self.person_history:
            prev_positions = self.person_history[person_id]
            
            if len(prev_positions) > 0:
                prev_distance = prev_positions[-1]
                
                # Vérifier si la personne a traversé (changement de signe)
                # On détecte quand la distance change de signe
                if (prev_distance < 0 and distance > 0):
                    # La ligne a été franchie vers la droite
                    direction = "→"
                    has_crossed = True
                    self.crossed_persons.add(person_id)
                    self.crossing_events.append({
                        'person_id': person_id,
                        'direction': direction,
                        'position': (center_x, center_y)
                    })
                elif (prev_distance > 0 and distance < 0):
                    # La ligne a été franchie vers la gauche
                    direction = "←"
                    has_crossed = True
                    self.crossed_persons.add(person_id)
                    self.crossing_events.append({
                        'person_id': person_id,
                        'direction': direction,
                        'position': (center_x, center_y)
                    })
        
        # Mettre à jour l'historique (garder seulement les 5 dernières positions)
        self.person_history[person_id].append(distance)
        if len(self.person_history[person_id]) > 5:
            self.person_history[person_id].pop(0)
        
        return has_crossed, direction
    
    def draw_line(self, frame, color=(0, 255, 0), thickness=2):
        """
        Dessine la ligne virtuelle sur le frame.
        
        Args:
            frame: Image sur laquelle dessiner
            color: Couleur de la ligne (BGR)
            thickness: Épaisseur de la ligne
            
        Returns:
            Frame annotée
        """
        p1 = tuple(self.line_start.astype(int))
        p2 = tuple(self.line_end.astype(int))
        
        # Dessiner la ligne
        cv2.line(frame, p1, p2, color, thickness)
        
        # Ajouter des petits cercles aux extrémités
        cv2.circle(frame, p1, 5, color, -1)
        cv2.circle(frame, p2, 5, color, -1)
        
        return frame
    
    def draw_crossing_events(self, frame, history_limit=20):
        """
        Affiche les événements de franchissement récents à droite de l'écran avec couleurs.
        
        Args:
            frame: Image à annoter
            history_limit: Nombre d'événements récents à afficher
            
        Returns:
            Frame annotée
        """
        # Afficher les événements récents à droite
        recent_events = self.crossing_events[-4:]  # Afficher seulement les 4 derniers
        
        y_offset = 30
        for event in recent_events:
            person_id = event['person_id']
            direction = event['direction']
            
            # Format : "ID X >" ou "ID X <"
            # Utiliser des caractères simples pour éviter les problèmes d'encodage
            direction_symbol = ">" if direction == "→" else "<"
            text = f"ID {person_id} {direction_symbol}"
            
            # Couleur basée sur la direction
            if direction == "→":
                color = (0, 165, 255)  # Orange pour droite
                text_color = (0, 0, 0)  # Texte noir
            else:  # "←"
                color = (255, 0, 0)    # Bleu pour gauche
                text_color = (255, 255, 255)  # Texte blanc
            
            # Calculer la position et la taille du texte
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            text_size = cv2.getTextSize(text, font, font_scale, thickness)
            text_width = text_size[0][0]
            text_height = text_size[0][1]
            
            # Position à droite avec marge
            margin = 15
            x = self.frame_width - text_width - margin
            y = y_offset + text_height
            
            # Dessiner un rectangle comme fond avec bordure
            cv2.rectangle(frame, 
                         (x - 10, y - text_height - 10), 
                         (self.frame_width - 5, y + 5), 
                         color, 
                         -1)  # Rempli
            
            # Ajouter une bordure noire
            cv2.rectangle(frame, 
                         (x - 10, y - text_height - 10), 
                         (self.frame_width - 5, y + 5), 
                         (0, 0, 0), 
                         2)  # Bordure noire
            
            # Afficher le texte
            cv2.putText(frame, text, (x, y),
                       font, font_scale, text_color, thickness)
            
            y_offset += 50
        
        return frame
    
    def get_crossing_count(self, direction=None):
        """
        Retourne le nombre de franchissements.
        
        Args:
            direction: "→" ou "←" pour filtrer par direction, None pour tous
            
        Returns:
            Nombre de franchissements
        """
        if direction is None:
            return len(self.crossing_events)
        
        return sum(1 for event in self.crossing_events if event['direction'] == direction)
    
    def get_stats(self):
        """
        Retourne les statistiques de franchissement.
        
        Returns:
            Dict avec statistiques
        """
        return {
            'total_crossings': len(self.crossing_events),
            'crossings_down': self.get_crossing_count("→"),
            'crossings_up': self.get_crossing_count("←"),
            'unique_persons_crossed': len(self.crossed_persons),
            'recent_events': self.crossing_events[-5:]  # Les 5 derniers événements
        }
    
    def reset(self):
        """
        Réinitialise le détecteur.
        """
        self.person_history.clear()
        self.crossed_persons.clear()
        self.crossing_events.clear()
