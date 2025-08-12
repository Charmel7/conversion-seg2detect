# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 10:02:35 2025

@author: Prosper Charmel
"""

from pathlib import Path
sourceItem=["train","valid","test"] # A adapter selon la stucture du dataset
source="D:/Dossier essai/Weed Maize Detection-2.v10i.yolov11"  # A adapter selon l'emplacement du dataset
ancienne_classes = {
0: 'maize',
1: 'weed',
2: 'maize',
3:'weed'
}# A adapter selon les classes presentes dans le dataset
nouvelle_classes = {
 'maize':1,
 'weed':0,
 }# A adapter selon les classes voulues dans le dataset
for item in sourceItem :
    try:
        chemin=source+"/"+item+"/labels"
        liste_fichier=Path(chemin).glob("*.txt")
        for fichier in liste_fichier  :
            with open (fichier.parent/fichier.name,"r") as f:
                lignes=f.readlines()
                nouvelles_lignes=[]
                for ligne in lignes:
                    ligne=ligne.strip()
                    chiffres=ligne.split(" ")
                    if (int(chiffres[0]) in list(ancienne_classes.keys())):
                        a=str(nouvelle_classes[ancienne_classes[int(chiffres[0])]])
                        chiffres[0]=a
                    nouvelles_lignes.append(" ".join(chiffres)+"\n")
            with open(fichier.parent/fichier.name,'w') as f:
                f.write(''.join(nouvelles_lignes))
    except:
        print("Un problème est survenu lors du remappage")
            
                
                    
                    