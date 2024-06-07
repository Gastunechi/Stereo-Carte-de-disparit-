import cv2
import numpy as np
import matplotlib.pyplot as plt

# Charger les images
img_left = cv2.imread('C:/Users/ADMIN/Documents/IFI/Vision par ordinateur/Tp3/Images/left.jpeg')
img_right = cv2.imread('C:/Users/ADMIN/Documents/IFI/Vision par ordinateur/Tp3/Images/right.jpeg')

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
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Convertir les keypoints en coordonnées
pts_left = np.float32([keypoints_left[m.queryIdx].pt for m in good_matches])
pts_right = np.float32([keypoints_right[m.trainIdx].pt for m in good_matches])

# Estimer la matrice fondamentale avec RANSAC
F, mask = cv2.findFundamentalMat(pts_left, pts_right, cv2.FM_RANSAC)

# Sélectionner les inliers
pts_left_inliers = pts_left[mask.ravel() == 1]
pts_right_inliers = pts_right[mask.ravel() == 1]

# Afficher la matrice fondamentale
print("Matrice fondamentale estimée (F) :")
print(F)

# Afficher les correspondances inliers
img_inliers = cv2.drawMatches(img_left, keypoints_left, img_right, keypoints_right, [good_matches[i] for i in range(len(good_matches)) if mask[i]], None, matchColor=(0, 255, 0), singlePointColor=None, flags=2)

plt.figure(figsize=(20, 10))
plt.imshow(img_inliers)
plt.title('Correspondances Inliers')
plt.axis('off')
plt.show()
