import cv2
import numpy as np
import matplotlib.pyplot as plt

# Charger les images
img_left = cv2.imread('C:/Users/ADMIN/Documents/IFI/Vision par ordinateur/Tp3/Images/left.jpeg', cv2.IMREAD_GRAYSCALE)
img_right = cv2.imread('C:/Users/ADMIN/Documents/IFI/Vision par ordinateur/Tp3/Images/right.jpeg', cv2.IMREAD_GRAYSCALE)

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
    if m.distance < 0.98 * n.distance:  # Utiliser un seuil plus strict
        good_matches.append(m)


# Convertir les keypoints en coordonnées
pts_left = np.float32([keypoints_left[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
pts_right = np.float32([keypoints_right[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Appliquer RANSAC pour trouver les inliers
H, mask = cv2.findHomography(pts_left, pts_right, cv2.RANSAC, 5.0)
matches_mask = mask.ravel().tolist()

# Compter le nombre de correspondances inliers trouvées
num_inliers = matches_mask.count(1)
print(f'Nombre de correspondances inliers trouvées : {num_inliers}')

# Dessiner les correspondances avec des lignes jaunes seulement pour les inliers
draw_params = dict(matchColor=(0, 255, 255),  # Couleur jaune pour les lignes
                   singlePointColor=None,
                   matchesMask=matches_mask,  # Dessiner seulement les inliers
                   flags=2)

img_matches = cv2.drawMatches(img_left, keypoints_left, img_right, keypoints_right, good_matches, None, **draw_params)

# Afficher les résultats
plt.imshow(img_matches)
plt.title('Correspondances SIFT (Lignes bleues) avec RANSAC')
plt.show()
