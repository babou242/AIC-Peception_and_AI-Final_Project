# 🔧 Guide de Troubleshooting - Problèmes de Tracking

## 🚨 Problèmes courants et solutions

### Problème 1 : Une même personne reçoit plusieurs IDs

**Symptôme :** Une personne se déplace et reçoit ID 0, puis ID 3, puis ID 7...

**Causes possibles :**
1. ✅ **Threshold trop élevé** - Le système ne reconnaît pas la même personne
2. ✅ **Embeddings de mauvaise qualité** - Crops trop petits ou flous
3. ✅ **Trop peu d'embeddings stockés** - Pas assez d'historique

**Solutions :**

#### Solution 1 : Baisser le threshold
```python
# Dans config.py
SIMILARITY_THRESHOLD = 0.60  # Au lieu de 0.70
MIN_CONFIDENCE = 0.55        # Au lieu de 0.65
```

#### Solution 2 : Augmenter l'historique
```python
# Dans config.py
MAX_EMBEDDINGS_PER_PERSON = 15  # Au lieu de 10
```

#### Solution 3 : Utiliser un modèle ReID plus puissant
```python
# Dans config.py
REID_MODEL = "osnet_x0_75"  # Au lieu de osnet_x0_25
```

#### Solution 4 : Améliorer la qualité des crops
```python
# Vérifier la résolution
CAMERA_WIDTH = 1280  # Au lieu de 640
CAMERA_HEIGHT = 720  # Au lieu de 480
```

---

### Problème 2 : Différentes personnes reçoivent le même ID

**Symptôme :** Personne A a ID 0, puis personne B entre et reçoit aussi ID 0

**Causes possibles :**
1. ✅ **Threshold trop bas** - Le système confond des personnes différentes
2. ✅ **Personnes trop similaires** - Vêtements identiques, même morphologie
3. ✅ **Timeout trop court** - L'ID expire et est réutilisé

**Solutions :**

#### Solution 1 : Augmenter le threshold
```python
# Dans config.py
SIMILARITY_THRESHOLD = 0.70  # Au lieu de 0.60
MIN_CONFIDENCE = 0.65        # Au lieu de 0.55
```

#### Solution 2 : Augmenter le timeout
```python
# Dans config.py
ID_TIMEOUT = 60.0  # Au lieu de 30.0
```

#### Solution 3 : Ne jamais réutiliser les IDs
Dans `src/database.py`, modifier la méthode `_cleanup_expired_ids()` :
```python
def _cleanup_expired_ids(self):
    """Supprime les IDs expirés mais ne réutilise jamais les numéros"""
    current_time = time.time()
    expired_ids = []
    
    for person_id, last_time in self.last_seen.items():
        if current_time - last_time > self.timeout:
            expired_ids.append(person_id)
    
    for person_id in expired_ids:
        del self.embeddings_history[person_id]
        del self.last_seen[person_id]
    
    # NE PAS réinitialiser self.next_id !
```

---

### Problème 3 : Performance faible (FPS < 10)

**Symptôme :** Le système est trop lent

**Solutions :**

#### Solution 1 : Utiliser un modèle plus léger
```python
# Dans config.py
REID_MODEL = "osnet_x0_25"  # Le plus rapide
YOLO_MODEL = "models/yolov8n.pt"  # Déjà le plus léger
```

#### Solution 2 : Réduire la résolution
```python
# Dans config.py
CAMERA_WIDTH = 480
CAMERA_HEIGHT = 360
```

#### Solution 3 : Activer CUDA (si GPU disponible)
```python
# Vérifier que PyTorch utilise le GPU
import torch
print(f"CUDA disponible: {torch.cuda.is_available()}")
```

---

### Problème 4 : Embeddings non normalisés

**Symptôme :** Erreurs ou similarités incorrectes

**Vérification :**
```python
import numpy as np

# Un embedding normalisé doit avoir une norme ≈ 1.0
norm = np.linalg.norm(embedding)
print(f"Norme: {norm}")  # Devrait être ≈ 1.0

if abs(norm - 1.0) > 0.01:
    print("⚠️  Embedding mal normalisé!")
```

**Solution :**
Dans `src/reid_extractor.py`, vérifier la normalisation :
```python
# Normalisation L2
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
norms[norms == 0] = 1  # Éviter division par zéro
embeddings = embeddings / norms
```

---

## 📊 Diagnostic avec les statistiques

### Interpréter les statistiques

```
FPS: 15.2
Personnes actuelles: 2
IDs actifs: 3
Nouveaux IDs: 8
Matches: 145
```

**Analyse :**
- **IDs actifs (3) > Personnes actuelles (2)** ✅ Normal (personnes sorties récemment)
- **Nouveaux IDs (8) trop élevé** ⚠️ Possible problème de faux négatifs
- **Ratio Matches/Nouveaux (145/8 = 18)** ✅ Bon ratio

### Ratios recommandés

| Métrique | Bon | Moyen | Mauvais |
|----------|-----|-------|---------|
| Matches/Nouveaux | >15 | 10-15 | <10 |
| IDs actifs/Personnes | 1-2x | 2-3x | >3x |
| FPS | >20 | 10-20 | <10 |

---

## 🧪 Tests de validation

### Test 1 : Stabilité d'ID

**Procédure :**
1. Une seule personne devant la caméra
2. Se déplacer dans la scène
3. Observer l'ID

**Résultat attendu :**
- ✅ ID reste constant (ex: toujours ID 0)
- ❌ ID change (ex: 0 → 2 → 5) = Problème

**Si échec :**
- Baisser `SIMILARITY_THRESHOLD`
- Augmenter `MAX_EMBEDDINGS_PER_PERSON`
- Utiliser `osnet_x0_5` ou plus

### Test 2 : Discrimination entre personnes

**Procédure :**
1. Personne A devant la caméra (ID 0)
2. Personne A sort
3. Personne B entre

**Résultat attendu :**
- ✅ Personne B reçoit ID 1 (nouveau)
- ❌ Personne B reçoit ID 0 (même que A) = Problème

**Si échec :**
- Augmenter `SIMILARITY_THRESHOLD`
- Augmenter `MIN_CONFIDENCE`
- Vérifier que les personnes sont visuellement différentes

### Test 3 : Réidentification après occlusion

**Procédure :**
1. Personne A devant la caméra (ID 0)
2. Personne A passe derrière un obstacle (pas vue)
3. Personne A réapparaît

**Résultat attendu :**
- ✅ Personne A retrouve ID 0
- ❌ Personne A reçoit ID 1 = Problème

**Si échec :**
- Augmenter `ID_TIMEOUT` (60s ou plus)
- Augmenter `MAX_EMBEDDINGS_PER_PERSON`

---

## 🎯 Configurations recommandées

### Configuration 1 : Précision maximale (peu de personnes)
```python
SIMILARITY_THRESHOLD = 0.70
MIN_CONFIDENCE = 0.65
MAX_EMBEDDINGS_PER_PERSON = 20
ID_TIMEOUT = 120.0
REID_MODEL = "osnet_x1_0"
```

**Avantages :** Très précis, peu de confusion  
**Inconvénients :** Lent, peut créer des doublons

### Configuration 2 : Équilibre (recommandé)
```python
SIMILARITY_THRESHOLD = 0.65
MIN_CONFIDENCE = 0.60
MAX_EMBEDDINGS_PER_PERSON = 10
ID_TIMEOUT = 30.0
REID_MODEL = "osnet_x0_5"
```

**Avantages :** Bon compromis vitesse/précision  
**Inconvénients :** Aucun majeur

### Configuration 3 : Performance maximale (foule)
```python
SIMILARITY_THRESHOLD = 0.60
MIN_CONFIDENCE = 0.55
MAX_EMBEDDINGS_PER_PERSON = 5
ID_TIMEOUT = 15.0
REID_MODEL = "osnet_x0_25"
```

**Avantages :** Très rapide, gère beaucoup de personnes  
**Inconvénients :** Plus de confusion possible

---

## 📝 Checklist de debug

- [ ] Vérifier que `torchreid` est installé
- [ ] Vérifier que les modèles se chargent sans erreur
- [ ] Vérifier les embeddings sont normalisés (norme ≈ 1.0)
- [ ] Tester avec 1 personne d'abord
- [ ] Observer les scores de confiance affichés
- [ ] Vérifier le ratio Matches/Nouveaux IDs
- [ ] Ajuster `SIMILARITY_THRESHOLD` par petits incréments (±0.05)
- [ ] Tester dans différentes conditions d'éclairage
- [ ] Vérifier que les crops ne sont pas trop petits

---

## 🆘 Si rien ne fonctionne

### Étape 1 : Vérifier l'installation
```bash
uv run python -c "import torchreid; print(torchreid.__version__)"
```

### Étape 2 : Tester l'extraction basique
```bash
uv run python examples/yolo_reid_tracking_example.py
```

### Étape 3 : Mode debug
Ajouter des prints dans `src/database.py` :
```python
def find_match(self, embedding, min_confidence=0.6):
    # ... code existant ...
    
    print(f"🔍 Debug Matching:")
    print(f"  Best ID: {best_id}, Similarity: {best_similarity:.4f}")
    print(f"  Threshold: {self.threshold}, Min Conf: {min_confidence}")
    
    # ... reste du code ...
```

### Étape 4 : Visualiser les similarités
Créer un script pour afficher la matrice de similarité entre tous les embeddings stockés.

---

**Besoin d'aide supplémentaire ?** Consultez les logs et statistiques en temps réel ! 📊
