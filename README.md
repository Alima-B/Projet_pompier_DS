# Projet_pompier_DS
**Scripts projets DS : prédiction des temps d'arrivé des pompiers de Londres**

L’objectif de ce projet est d’analyser et/ou d’estimer les temps de réponse et de mobilisation de la Brigade des Pompiers de Londres.
La brigade des pompiers de Londres est le service d'incendie et de sauvetage le plus actif du Royaume-Uni et l'une des plus grandes organisations de lutte contre l'incendie et de sauvetage au monde.

Le premier jeu de données fourni contient les détails de chaque incident traité depuis janvier 2009. Des informations sont fournies sur la date et le lieu de l'incident ainsi que sur le type d'incident traité.
https://data.london.gov.uk/dataset/london-fire-brigade-incident-records 

Le second jeu de données contient les détails de chaque camion de pompiers envoyé sur les lieux d'un incident depuis janvier 2009. Des informations sont fournies sur l'appareil mobilisé, son lieu de déploiement et les heures d'arrivée sur les lieux de l'incident.
https://data.london.gov.uk/dataset/london-fire-brigade-mobilisation-records

Ce projet a exploré différentes approches de modélisation, allant des régressions
linéaires aux algorithmes de classification en passant par des méthodes d'ensemble comme les
forêts aléatoires et le Boosting ainsi que du Deep Learning.

**Performance des modèles**

**Regression** : Erreur moyenne absolue de 60 secondes.    
**Classification** : Accuracy de 60% (avec des erreurs faites les valeurs à la limite de deux classes)

La distance entre la casernes intervenant et le lieu de l'incendie s'est avérée être la variable la plus importante dans les prédictions
