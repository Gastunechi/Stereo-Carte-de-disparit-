import cv2
import matplotlib.pyplot as plt

# Chemins des fichiers images
chemin_image_gauche = 'C:/Users/ADMIN/Documents/IFI/Vision par ordinateur/Tp3/Images/Solution_code/Figure_050.png'
chemin_image_droite = 'C:/Users/ADMIN/Documents/IFI/Vision par ordinateur/Tp3/Images/Solution_code/Inversed/Figure_050.png'

# Charger les images en couleur
image_gauche = cv2.imread(chemin_image_gauche)
image_droite = cv2.imread(chemin_image_droite)

# Vérifier si les images sont chargées correctement
if image_gauche is None:
    print(f"Erreur de chargement de l'image gauche : {chemin_image_gauche}")
if image_droite is None:
    print(f"Erreur de chargement de l'image droite : {chemin_image_droite}")

# Afficher les images côte à côte
if image_gauche is not None and image_droite is not None:
    # Convertir les images de BGR à RGB pour l'affichage avec matplotlib
    image_gauche = cv2.cvtColor(image_gauche, cv2.COLOR_BGR2RGB)
    image_droite = cv2.cvtColor(image_droite, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image_gauche)
    plt.title('Image Gauche')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image_droite)
    plt.title('Image Droite')
    plt.axis('off')

    plt.show()
else:
    print("Une des images n'a pas pu être chargée.")
