import cv2
import numpy as np
import matplotlib.pyplot as plt

# Charger les images
img_left = cv2.imread('C:/Users/ADMIN/Documents/IFI/Vision par ordinateur/Tp3/Images/gauche.jpeg', cv2.IMREAD_GRAYSCALE)
img_right = cv2.imread('C:/Users/ADMIN/Documents/IFI/Vision par ordinateur/Tp3/Images/droite.jpeg', cv2.IMREAD_GRAYSCALE)

# Initialiser le détecteur de points d'intérêt SIFT
sift = cv2.SIFT_create()

# Calculer les points d'intérêt SIFT et les descripteurs
keypoints_left, descriptors_left = sift.detectAndCompute(img_left, None)
keypoints_right, descriptors_right = sift.detectAndCompute(img_right, None)

# Utiliser le matcher BFMatcher pour faire la mise en correspondance
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors_left, descriptors_right, k=2)

# Appliquer le ratio test de David Lowe pour sélectionner les bons matches
good_matches = []
for m, n in matches:
    if m.distance < 0.95 * n.distance:  # Utiliser un seuil plus strict
        good_matches.append(m)

# Convertir les keypoints en coordonnées
pts_left = np.float32([keypoints_left[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
pts_right = np.float32([keypoints_right[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

# Heuristique 1: Contraintes géométriques
def geometric_constraints(good_matches, pts_left, pts_right, max_horizontal_diff=10.0):
    filtered_matches = []
    for i, (pt_left, pt_right) in enumerate(zip(pts_left, pts_right)):
        # Vérifier la contrainte horizontale
        if abs(pt_left[1] - pt_right[1]) <= max_horizontal_diff:
            filtered_matches.append(good_matches[i])
    return filtered_matches

good_matches = geometric_constraints(good_matches, pts_left, pts_right)

# Mettre à jour les coordonnées après la première heuristique
pts_left = np.float32([keypoints_left[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
pts_right = np.float32([keypoints_right[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

# Heuristique 2: Distance des correspondances
def distance_constraints(good_matches, pts_left, pts_right, min_dist=5.0, max_dist=50.0):
    filtered_matches = []
    for i, (pt_left, pt_right) in enumerate(zip(pts_left, pts_right)):
        dist = np.linalg.norm(pt_left - pt_right)
        if min_dist <= dist <= max_dist:
            filtered_matches.append(good_matches[i])
    return filtered_matches

good_matches = distance_constraints(good_matches, pts_left, pts_right)

# Mettre à jour les coordonnées après la deuxième heuristique
pts_left = np.float32([keypoints_left[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
pts_right = np.float32([keypoints_right[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

# Heuristique 3: Vérification croisée
def cross_checking(good_matches, descriptors_left, descriptors_right):
    reverse_matches = bf.knnMatch(descriptors_right, descriptors_left, k=2)
    reverse_good_matches = []
    for m, n in reverse_matches:
        if m.distance < 0.6 * n.distance:
            reverse_good_matches.append(m)
    
    final_matches = []
    for match in good_matches:
        left_idx = match.queryIdx
        right_idx = match.trainIdx
        for rev_match in reverse_good_matches:
            if rev_match.queryIdx == right_idx and rev_match.trainIdx == left_idx:
                final_matches.append(match)
                break
    return final_matches

good_matches = cross_checking(good_matches, descriptors_left, descriptors_right)

# Dessiner les correspondances avec des lignes jaunes
img_matches = np.array(cv2.drawMatches(img_left, keypoints_left, img_right, keypoints_right, good_matches, None, matchColor=(0, 255, 255), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS))

# Afficher les résultats
plt.imshow(img_matches)
plt.title('Correspondances SIFT (Lignes bleues) avec heuristiques')
plt.show()


# Calcul de la matrice fondamentale à partir des points correspondants
F, mask = cv2.findFundamentalMat(pts_left, pts_right, cv2.FM_RANSAC)

# Sélectionner uniquement les inliers
pts_left_inliers = pts_left[mask.ravel() == 1]
pts_right_inliers = pts_right[mask.ravel() == 1]

# Affichage des points inliers
plt.scatter(pts_left_inliers[:, 0], pts_left_inliers[:, 1], c='r', marker='o')
plt.scatter(pts_right_inliers[:, 0], pts_right_inliers[:, 1], c='b', marker='x')
plt.title('Inliers après estimation de la matrice fondamentale')
plt.show()