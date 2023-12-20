# Explorer les possiblités de l'enquête emploi

Groupe Moisan, Guillon, Barrau

But: Ce projet Python cherche à prédire le fait d'être en emploi à partir des caractéristiques socio-démographiques, en utilisant les enquêtes emploi de 2019 et 2020

## Structure du Projet

### I -  Statistiques descriptives
Dans la première section, 


### II - Réduction de Dimensionnalité avec PCA
Dans cette seconde section, l'Analyse en Composantes Principales (PCA) est appliquée aux caractéristiques socio-démographiques que sont  ... .  Un graphique présente ainsi la variance expliquée par chaque composante principale.

 ### III - Modelisation

Dans un premier temps, deux approches de modélisation sont explorées pour prédire le statut d'emploi: Un modèle SVM et une régression logistique. Une étape de recherche des hyperparamètres optimaux est effectuée pour chaque algorithme, suivie d'une évaluation des performances à l'aide de matrices de confusion. Comme la régression logistique permet de déterminer facilement le nombre de variables pertinentes avec la régularisation "l1", nous continuerons à travailler avec ce modèle par la suite. 

Dans un second temps, un modèle linéaire Lasso avec sélection croisée du paramètre de régularisation est mis en œuvre pour sélectionner un sous-ensemble des variables les plus influentes. Le nombre de caractéristiques sélectionnées est limité à 10 pour observer si la performance de prediction diminue.

Enfin, reconnaissant les défis liés à la prédiction du statut d'emploi, en particulier la distinction entre les chômeurs de courte durée et les actifs occupés, le projet tente de prédire le statut "actif" . Cela implique la construction d'un prédicteur pour être "actif" et la comparaison de la qualité de la prédiction avec les résultats précédents.
  
## Utilisation

Pour reproduire les résultats, suivez les étapes décrites dans le notebook Jupyter fourni. N'hésitez pas à adapter et étendre ce projet pour explorer d'autres questions ou améliorer la prédiction en utilisant différentes approches de ML.


