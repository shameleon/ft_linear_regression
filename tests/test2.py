import sys


def success():
    print(sys.argv[1])


if __name__ == "__main__":
    if (len(sys.argv) == 2):
        success()

"""
La liste des arguments de la ligne de commande passés à un script Python.
argv[0] est le nom du script (chemin complet, ou non, en fonction
du système d'exploitation). Si la commande a été exécutée avec l'option
-c de l'interpréteur, argv[0] vaut la chaîne '-c'.
Si aucun nom de script n'a été donné à l'interpréteur Python, argv[0]
 sera une chaîne vide.
"""
