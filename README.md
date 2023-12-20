# Explorer les possiblités de l'enquête emploi

Groupe Moisan, Guillon, Barrau

But: Ce projet Python se concentre sur la prédiction du statut d'emploi en se basant sur les caractéristiques socio-démographiques, en utilisant les enquêtes sur l'emploi en France de 2019 et 2020. Le projet comporte plusieurs étapes clés, dont l'Analyse en Composantes Principales (PCA) pour la réduction de dimensionnalité et l'application de modèles à Support Vector Machine (SVM) et de Régression Logistique pour la classification.

## Structure du Projet

### I -  Statistiques descriptives



### II - Réduction de Dimensionnalité avec PCA
Dans la première section, l'Analyse en Composantes Principales (PCA) est appliquée aux caractéristiques socio-démographiques à l'aide de la classe PCA de la bibliothèque scikit-learn. Le scree plot est visualisé pour montrer la variance expliquée par chaque composante principale.

 ### III - Modelisation

#### A - Support Vector Machine   
  Le modèle SVM est mis en œuvre pour prédire le statut d'emploi. Le projet explore l'impact du paramètre de régularisation (C) sur les performances du modèle en utilisant une plage de     valeurs. La méthode GridSearchCV est ensuite utilisée pour ajuster finement le paramètre de régularisation, aboutissant à un modèle SVM optimisé.

#### B - Régression Logistique
  Le projet examine également l'utilisation de la Régression Logistique pour prédire le statut d'emploi. Similaire à la section SVM, le paramètre de régularisation (C) est varié pour       observer son effet sur la précision du modèle. GridSearchCV est à nouveau utilisé pour trouver la combinaison optimale d'hyperparamètres pour la Régression Logistique.
  
#### C - Sélection de Variables avec Lasso
  Un modèle linéaire Lasso avec sélection croisée du paramètre de régularisation est mis en œuvre pour sélectionner un sous-ensemble des variables les plus influentes. Le nombre de         caractéristiques sélectionnées est limité à 10 pour éviter le surajustement.

#### D - Prédiction Actif/Inactif
  Reconnaissant les défis liés à la prédiction du statut d'emploi, en particulier la distinction entre les chômeurs de courte durée et les actifs occupés, le projet tente de prédire le     statut "actif" séparément. Cela implique la construction d'un prédicteur pour être "actif" et la comparaison de la qualité de la prédiction.
  
## Utilisation

Pour reproduire les résultats, suivez les étapes décrites dans le notebook Jupyter fourni. N'hésitez pas à reprendre et à étendre ce projet pour explorer d'autres questions ou améliorer la prédiction.
