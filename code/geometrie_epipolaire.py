import cv2
import numpy as np
import matplotlib.pyplot as plt

# Chemins des fichiers images
chemin_image_gauche = 'C:/Users/ADMIN/Documents/IFI/Vision par ordinateur/Tp3/Images/left.jpeg'
chemin_image_droite = 'C:/Users/ADMIN/Documents/IFI/Vision par ordinateur/Tp3/Images/right.jpeg'
# Charger les images stéréoscopiques
image_gauche = cv2.imread(chemin_image_gauche)
image_droite = cv2.imread(chemin_image_droite)

# Vérifier si les images sont chargées correctement
if image_gauche is None:
    print(f"Erreur de chargement de l'image gauche : {chemin_image_gauche}")
if image_droite is None:
    print(f"Erreur de chargement de l'image droite : {chemin_image_droite}")

# Continuer seulement si les deux images sont chargées correctement
if image_gauche is not None and image_droite is not None:
    # Conversion en niveaux de gris
    img1 = cv2.cvtColor(image_gauche, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(image_droite, cv2.COLOR_BGR2GRAY)

    # Créer l'objet SIFT avec des paramètres ajustés
    sift = cv2.SIFT_create()

    # Trouver les points clés et descripteurs avec SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Créer un BFMatcher avec distance euclidienne
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # Trouver les deux meilleurs matches pour chaque descripteur
    matches = bf.knnMatch(des1, des2, k=2)

    # Appliquer le test de ratio de Lowe pour filtrer les bons matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.98 * n.distance:  # Utiliser 0.75 comme ratio pour inclure plus de matches
            good_matches.append(m)

    # Filtrage supplémentaire basé sur la distance des pixels ou l'alignement horizontal
    filtered_matches = []
    for match in good_matches:
        # Obtenir les coordonnées des points correspondants
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        # Filtrer selon la distance entre les points
        if abs(y1 - y2) < 2:  # Seuil de 2 pixels de différence verticale
            filtered_matches.append(match)

    # Préparer les listes de points pour la matrice fondamentale
    pts1 = np.float32([kp1[m.queryIdx].pt for m in filtered_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in filtered_matches])

    # Visualiser les matches filtrés
    img_filtered_matches = cv2.drawMatches(img1, kp1, img2, kp2, filtered_matches, None, matchColor=(255, 255,0 ), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.figure(figsize=(15, 10))
    plt.imshow(img_filtered_matches)
    plt.title('Correspondances filtrées')
    plt.axis('off')
    plt.show()

    # Calcul de la matrice fondamentale avec RANSAC
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

    if mask is not None:
        # Sélectionner les inliers correspondants au masque renvoyé par findFundamentalMat
        pts1_inliers = pts1[mask.ravel() == 1]
        pts2_inliers = pts2[mask.ravel() == 1]

        # Affichage des points inliers et des correspondances
        img_inlier_matches = cv2.drawMatches(img1, kp1, img2, kp2, [filtered_matches[i] for i in range(len(filtered_matches)) if mask[i]], None, matchColor=(0, 255, 255), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Fonction pour afficher les images avec les correspondances des inliers
        def display_inliers(image):
            plt.figure(figsize=(15, 10))
            plt.imshow(image)
            plt.title('Correspondances des Inliers')
            plt.axis('off')
            plt.show()

        display_inliers(img_inlier_matches)
        
        # Afficher la matrice fondamentale
        print("Matrice fondamentale estimée (F) :")
        print(F)

        # Afficher le nombre de correspondances avant et après le filtrage
        print(f"Nombre de correspondances avant filtrage: {len(good_matches)}")
        print(f"Nombre de correspondances après filtrage supplémentaire: {len(filtered_matches)}")
        print(f"Nombre d'inliers après RANSAC: {np.sum(mask)}")
    else:
        print("Impossible de calculer la matrice fondamentale, pas assez de correspondances valides.")
else:
    print("Chargement des images échoué. Vérifiez les chemins des fichiers et réessayez.")
