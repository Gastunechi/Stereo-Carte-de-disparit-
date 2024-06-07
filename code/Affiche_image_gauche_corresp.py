import cv2
import numpy as np
from matplotlib import pyplot as plt

# Lecture des deux images
imgL = cv2.imread('C:/Users/ADMIN/Documents/IFI/Vision par ordinateur/Tp3/Images/left.jpeg', cv2.IMREAD_GRAYSCALE)  # Image gauche
imgR = cv2.imread('C:/Users/ADMIN/Documents/IFI/Vision par ordinateur/Tp3/Images/right.jpeg', cv2.IMREAD_GRAYSCALE)  # Image droite

# Détection des points d'intérêt
sift = cv2.SIFT_create()
kpL, desL = sift.detectAndCompute(imgL, None)
kpR, desR = sift.detectAndCompute(imgR, None)

# Correspondances des points
bf = cv2.BFMatcher()
matches = bf.knnMatch(desL, desR, k=2)

# Appliquer le ratio test de Lowe pour filtrer les bonnes correspondances
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

# Trier les correspondances par distance
good = sorted(good, key=lambda x: x.distance)
# Conserver les 157 correspondances les plus fortes
strong_matches = good[:200]

# Préparer les points pour dessiner les lignes
ptsL = np.float32([kpL[m.queryIdx].pt for m in strong_matches])
ptsR = np.float32([kpR[m.trainIdx].pt for m in strong_matches])

# Dessiner les correspondances les plus fortes sur l'image gauche
output_img = cv2.cvtColor(imgL, cv2.COLOR_GRAY2BGR)
for ptL, ptR in zip(ptsL, ptsR):
    ptL = (int(ptL[0]), int(ptL[1]))
    ptR = (int(ptR[0]), int(ptR[1]))
    # Ajuster la longueur des lignes pour qu'elles soient plus courtes et dirigées vers la droite
    line_length = 35
    direction = (line_length, 0)  # Direction vers la droite
    end_point = (ptL[0] + direction[0], ptL[1] + direction[1])
    output_img = cv2.circle(output_img, ptL, 5, ( 255, 255, 0), -1)  # Points en jaune
    output_img = cv2.line(output_img, ptL, end_point, (255, 255, 0), 2)  # Lignes en jaune

# Affichage de l'image avec les correspondances
plt.figure(figsize=(10, 7))
plt.imshow(output_img)
plt.title('Correspondances les plus fortes (200) - Image gauche')
plt.axis('off')
plt.show()