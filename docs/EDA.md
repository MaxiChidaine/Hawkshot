# PLAN EDA

## 01_dataset_overview

Question centrale : comment le dataset FD001 est-il structuré et quelles variables sont exploitables ?

Objective and scope
Data loading
Dataset structure and integrity
Engine lifetime distribution
Constant and near-constant variables
Variables retained for further analysis
Main findings and next steps
Sortie attendue

À la fin, nous devons connaître :

le nombre de moteurs et d’observations ;
la signification des lignes et colonnes ;
les durées de vie des moteurs ;
la présence éventuelle de données manquantes ou dupliquées ;
les variables retirées ;
les capteurs transmis au notebook suivant.

Aucune conclusion approfondie sur la qualité prédictive des capteurs n’est encore attendue.

## 02_sensor_trend_analysis

Question centrale : quels capteurs suivent une évolution liée au vieillissement des moteurs, et cette évolution est-elle cohérente dans la flotte ?

Individual engine exploration
Comparison across selected engines
Per-engine Spearman correlations
Fleet-level correlation summary
Sensor consistency and ranking
Detailed analysis of selected sensors
Main findings
Sortie attendue

Pour chaque capteur :

100 coefficients de Spearman ;
une direction dominante ;
une intensité médiane ;
une mesure de cohérence entre moteurs ;
des visualisations permettant d’interpréter le résultat.

## 03_rul_target_and_features

Question centrale : comment transformer les trajectoires en données adaptées à la prédiction de la RUL ?

calcul de la RUL ;
comparaison RUL linéaire / RUL plafonnée ;
création des premières features ;
préparation des jeux par moteur ;
prévention des fuites de données.

## 04_baseline_models

Question centrale : quelle performance minimale obtient-on avec des modèles simples et reproductibles ?

baseline naïve ;
régression linéaire ;
modèle arbre ou ensemble ;
métriques ;
validation par groupes de moteurs.
05_model_evaluation

Question centrale : quels modèles et quelles variables généralisent réellement ?

comparaison des modèles ;
erreurs selon la phase de vie ;
importance des variables ;
analyse des moteurs difficiles ;
résultats finaux.

