#Ce script reprends les principales étapes de notre notebook sur la partie modélisation, sous partie D portant sur la sélection de paramètre (modèle lassso)
#Simplement, le choix est fait ici de fit sur la base entière (soit en divisant train et test en 2019, soit en utilisant 2019 en train et 2020 en test), et non sur le subsample
#La partie de recherche du paramètre de régularisation, comme des meilleurs paramètres, est faites sur le subsample pour maintenir un temps de calcul acceptable

import pandas as pd
import numpy as np
import requests
from io import BytesIO #permet de stocker en mémoire
from zipfile import ZipFile
import matplotlib.pyplot as plt

#Imports pour la modélisation
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel

#import des données
url2019 = "https://www.insee.fr/fr/statistiques/fichier/4809583/fd_eec19_csv.zip" #enquete 2019
url2020="https://www.insee.fr/fr/statistiques/fichier/5393560/fd_eec20_csv.zip" #enquête 2020 en exemple
requete = requests.get(url2019)
zip_df = ZipFile(BytesIO(requete.content)) #créer un fichier ZIP
with zip_df.open(zip_df.namelist()[0]) as extrait:
    EEC_2019 = pd.read_csv(extrait, delimiter=";") # Lire le fichier CSV avec pandas
requete = requests.get(url2020)
zip_df = ZipFile(BytesIO(requete.content)) #créer un fichier ZIP
with zip_df.open(zip_df.namelist()[0]) as extrait:
    EEC_2020 = pd.read_csv(extrait, delimiter=";") # Lire le fichier CSV avec pandas

#conservation uniquement de nos variables
list_var_selected = ["ACTEU","ANNEE" ,"TRIM", "AGE3" ,  "AGE5"  , "CATAU2010R" ,
"COURED" ,"CSA" ,"CSP" , "DIP11","ENFRED" , "METRODOM" , "NFRRED" , "SEXE" , "TYPMEN7"] 
EEC_2019 = EEC_2019[list_var_selected]
EEC_2020 = EEC_2020[list_var_selected] 
list_var = list(EEC_2019.columns.values)

#recodage de la PCS
EEC_2019['PCS'] = EEC_2019['CSA'].add(EEC_2019['CSP'], fill_value=0)
EEC_2020['PCS'] = EEC_2020['CSA'].add(EEC_2020['CSP'], fill_value=0)
EEC_2019 = EEC_2019.drop(['CSA', 'CSP'], axis=1)
EEC_2020 = EEC_2020.drop(['CSA', 'CSP'], axis=1)
EEC_2019 = EEC_2019[EEC_2019['PCS'] != 10]

#exclusion des NA
EEC_2019 = EEC_2019.dropna() 
EEC_2020 = EEC_2020.dropna() 

# Conversion de l'ensemble des variables catégorielles en indicatrices
EEC_2019 = pd.get_dummies(EEC_2019, columns=["AGE3" ,  "AGE5"  , "CATAU2010R" ,
"PCS", "DIP11", "NFRRED" , "TYPMEN7"])
EEC_2020 = pd.get_dummies(EEC_2020, columns=["AGE3" ,  "AGE5"  , "CATAU2010R" ,
"PCS" , "DIP11", "NFRRED" , "TYPMEN7"])
EEC_2019['EMPLOI'] = EEC_2019['ACTEU'].apply(lambda x: x == 1)
EEC_2020['EMPLOI'] = EEC_2020['ACTEU'].apply(lambda x: x == 1)
EEC_2019['ACTIF'] = EEC_2019['ACTEU'].apply(lambda x: (x == 1) or (x == 2))
EEC_2020['ACTIF'] = EEC_2020['ACTEU'].apply(lambda x: (x == 1) or (x == 2))
EEC_2019['FEMME'] = EEC_2019['SEXE'].apply(lambda x: x == 2)
EEC_2020['FEMME'] = EEC_2020['SEXE'].apply(lambda x: x == 2)
EEC_2019['COUPLE'] = EEC_2019['COURED'].apply(lambda x: x == 2)
EEC_2020['COUPLE'] = EEC_2020['COURED'].apply(lambda x: x == 2)
EEC_2019['ENFANT'] = EEC_2019['ENFRED'].apply(lambda x: x == 2)
EEC_2020['ENFANT'] = EEC_2020['ENFRED'].apply(lambda x: x == 2)
EEC_2019['DOM'] = EEC_2019['METRODOM'].apply(lambda x: x == 2)
EEC_2020['DOM'] = EEC_2020['METRODOM'].apply(lambda x: x == 2)
EEC_2019 = EEC_2019.drop(['METRODOM', 'ENFRED' , 'COURED', 'SEXE',"ACTEU"], axis=1)
EEC_2020 = EEC_2020.drop(['METRODOM', 'ENFRED' , 'COURED', 'SEXE',"ACTEU"], axis=1)

#On prends un sous-ensemble
EEC_2019_subsample = EEC_2019.sample(n=10000, random_state=3) 

#Création de nos arrays de train et de test
X = np.array(EEC_2019_subsample.drop(columns=["ACTIF","EMPLOI","ANNEE", "TRIM"]))
y = np.array(EEC_2019_subsample["EMPLOI"])
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=.2, random_state=4)

#création de nos array de features et variable d'intérêt avec l'ensemble de la base, ainsi qu'une séparation train / test
X_total = np.array(EEC_2019.drop(columns=["ACTIF","EMPLOI","ANNEE", "TRIM"]))
y_total = np.array(EEC_2019["EMPLOI"])
X_train_total, X_test_total, y_train_total, y_test_total = train_test_split(X_total, 
                                                    y_total, 
                                                    test_size=.2, random_state=4)

#création des arrays pour 2020
X_2020 = np.array(EEC_2020.drop(columns=["ACTIF","EMPLOI","ANNEE", "TRIM"]))
y_2020 = np.array(EEC_2020["EMPLOI"])

## Sélection de variable par lasso

#  Lasso linear model with cross-validated selection of the regularization parameter (alpha) to find the optimal
clf = LassoCV()
clf.fit(X, y)

# Imposition de 10 features maximum
sfm = SelectFromModel(clf, prefit=False, max_features=10)

# X transform contient un nombre réduit de caractéristiques basé sur la sélection
# des caractéristiques par régression Lasso
sfm.fit(X,y)
X_transform = sfm.transform(X)

#recherche de la mailleure performance sur le subsample
params= { 'C':np.linspace(0.001, 1, 50), 'tol': [0.01, 0.1, 1, 10] }
gs = GridSearchCV(estimator=LogisticRegression( random_state=3, penalty = "l1", solver='liblinear'), 
                   param_grid=params,
                   cv=5)
gs.fit(X_train, y_train)

clf = LogisticRegression(tol=gs.best_params_['tol'], C=gs.best_params_['C'],
                         random_state=3, penalty="l1", solver='liblinear')

#fit sur total et matrice de confusion
#On essaie ici de regarder ce que cela donne sur l'ensemble de la base 2019 dès lors que l'on prends en compte l'ensemble des observations
#En partant du choix du paramètre de régularisation trouvé précédemment.

clf.fit(X_train_total, y_train_total)
y_pred_total = clf.predict(X_test_total)

cm= confusion_matrix(y_test_total, y_pred_total)
disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=clf.classes_
       )
disp.plot()
plt.savefig("Scripts/Results/Selection/Confusion_Matrix_2019.png")

#fit sur 2019 précision 2020
clf.fit(X_total, y_total)
y_pred_2020 = clf.predict(X_2020)
cm= confusion_matrix(y_2020, y_pred_2020)
disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=clf.classes_
       )
disp.plot()
plt.savefig("Scripts/Results/Selection/Confusion_Matrix_2020_fit_2019")