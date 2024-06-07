import cv2
import numpy as np
import matplotlib.pyplot as plt

# Charger les images
img_left = cv2.imread('C:/Users/ADMIN/Documents/IFI/Vision par ordinateur/Tp3/Images/right.jpeg', cv2.IMREAD_GRAYSCALE)
img_right = cv2.imread('C:/Users/ADMIN/Documents/IFI/Vision par ordinateur/Tp3/Images/left.jpeg', cv2.IMREAD_GRAYSCALE)

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
    if m.distance < 0.50 * n.distance:
        good_matches.append(m)

# Compter le nombre de correspondances trouvées
num_matches = len(good_matches)
print(f'Nombre de correspondances trouvées : {num_matches}')

# Dessiner les correspondances sur l'image gauche
img_matches = np.array(cv2.drawMatches(img_left, keypoints_left, img_right, keypoints_right, good_matches, None, matchColor=( 255, 255, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS))

# Afficher les résultats
plt.imshow(img_matches)
plt.title('Correspondances SIFT')
plt.show()



