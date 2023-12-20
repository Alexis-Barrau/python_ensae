# python_ensae
Prédiction du Statut d'Emploi à l'aide des Caractéristiques Socio-démographiques
Groupe Moisan, Guillon, Barrau

But: Ce projet Python se concentre sur la prédiction du statut d'emploi en se basant sur les caractéristiques socio-démographiques, en utilisant les enquêtes sur l'emploi en France de 2019 et 2020. Le projet comporte plusieurs étapes clés, dont l'Analyse en Composantes Principales (PCA) pour la réduction de dimensionnalité et l'application de modèles à Support Vector Machine (SVM) et de Régression Logistique pour la classification.

Structure du Projet
1. Statistiques descriptives

1. Réduction de Dimensionnalité avec PCA
Dans la première section, l'Analyse en Composantes Principales (PCA) est appliquée aux caractéristiques socio-démographiques à l'aide de la classe PCA de la bibliothèque scikit-learn. Le scree plot est visualisé pour montrer la variance expliquée par chaque composante principale.

3. Modelisation

  A. Support Vector Machine   
  Le modèle SVM est mis en œuvre pour prédire le statut d'emploi. Le projet explore l'impact du paramètre de régularisation (C) sur les performances du modèle en utilisant une plage de     valeurs. La méthode GridSearchCV est ensuite utilisée pour ajuster finement le paramètre de régularisation, aboutissant à un modèle SVM optimisé.

  B. Régression Logistique
  Le projet examine également l'utilisation de la Régression Logistique pour prédire le statut d'emploi. Similaire à la section SVM, le paramètre de régularisation (C) est varié pour       observer son effet sur la précision du modèle. GridSearchCV est à nouveau utilisé pour trouver la combinaison optimale d'hyperparamètres pour la Régression Logistique.
  
  C. Sélection de Variables avec Lasso
  Un modèle linéaire Lasso avec sélection croisée du paramètre de régularisation est mis en œuvre pour sélectionner un sous-ensemble des variables les plus influentes. Le nombre de         caractéristiques sélectionnées est limité à 10 pour éviter le surajustement.

  D. Prédiction Actif/Inactif
  Reconnaissant les défis liés à la prédiction du statut d'emploi, en particulier la distinction entre les chômeurs de courte durée et les actifs occupés, le projet tente de prédire le     statut "actif" séparément. Cela implique la construction d'un prédicteur pour être "actif" et la comparaison de la qualité de la prédiction.

Résultats du Projet
Performance du SVM : Le modèle SVM, optimisé via GridSearchCV, atteint une précision de test de 89,6%. Cependant, il y a une indication de surajustement, car la performance diminue sur l'ensemble de test pour de petites valeurs du paramètre de régularisation (C).

Performance de la Régression Logistique : Le modèle de Régression Logistique, malgré un schéma de performance étrange, atteint une précision d'environ 86,5% sur l'ensemble de test. GridSearchCV aide à ajuster finement les hyperparamètres, et la performance du modèle est légèrement inférieure à celle du modèle SVM.

Sélection de Variables : La régression Lasso est utilisée pour la sélection de variables, et le nombre de variables est limité à 10. Le modèle de Régression Logistique utilisant ces variables sélectionnées maintient une précision raisonnablement élevée mais avec un nombre réduit de caractéristiques.

Prédiction Actif/Inactif : Une prédiction distincte pour le statut "actif" montre une performance améliorée, réduisant les erreurs de prédiction de 269 à 209 par rapport à la prédiction directe du statut d'emploi.

Utilisation
Pour reproduire les résultats, suivez les étapes décrites dans le notebook Jupyter fourni. Assurez-vous que les bibliothèques requises, telles que scikit-learn, numpy, pandas et matplotlib, sont installées dans votre environnement Python.

N'hésitez pas à adapter et étendre ce projet pour explorer des fonctionnalités supplémentaires, affiner davantage les modèles ou utiliser d'autres techniques d'apprentissage automatique pour de meilleures prédictions.
