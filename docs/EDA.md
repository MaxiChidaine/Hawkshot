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

Comment transformer les trajectoires run-to-failure de FD001 en un jeu de données supervisé, exploitable pour prédire la RUL sans introduire de fuite d’information ?

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

Déroulement des expériences : 

EXPÉRIENCE 0
cycle only
→ combien l'âge seul permet-il de prédire ?


EXPÉRIENCE 1
14 raw sensors
→ baseline capteurs


EXPÉRIENCE 2
14 raw sensors + cycle
→ valeur ajoutée de l'âge


EXPÉRIENCE 3
raw sensors + temporal features
→ valeur ajoutée de la dynamique


EXPÉRIENCE 4
variation des fenêtres temporelles
→ quelle quantité d'historique est utile ?


EXPÉRIENCE 5
sensor ablations
→ contribution complémentaire de chaque capteur


EXPÉRIENCE 6
group ablations / feature subsets
→ tester les hypothèses issues de l'EDA


