# OC_Project7
## Projet OpenClassroom parcours Data Scientist

### Description du projet
L'organisme de prêt "Prêt à dépenser" souhaite mettre en place une interface à destination des conseillers clientèle, afin de les aider à décider d'accorder ou non un prêt à un client donné.  
Dans cette perspective, il s'agit de mettre en place **un modèle de scoring** permettant de décider, en fonction des informations connues du client, s'il existe un risque suffisamment fort d'insolvabilité pour refuser le prêt.  
Il doit être tenu compte du fait que le coût d'un prêt accordé à un "mauvais payeur" est significativement plus grand (au moins 100 fois) que le coût d'un prêt refusé à un "bon payeur".  
Afin d'élaborer ce modèle de scoring, un dataset rassemblant des informations de plus de 300 000 clients est mis à disposition. 122 indicateurs sont plus ou moins renseignés selon les clients et l'information du défaut de paiement est connue. Les clients ayant été en défaut de paiement sont nettement moins nombreux, ils représentent environ 8% de l'ensemble des clients.  
Une fois le modèle défini, il s'agit de l'appeler depuis **une API (back-end)** qui doit renvoyer au conseiller clientèle l'ensemble des informations dont il a besoin pour **prendre une décision et l'expliquer**.  
Il a accès à **une application (front-end)** qui ne contient pas de modèle enregistré ni d'information relative aux clients références ou nouveaux, seules des requêtes API permettent d'obtenir les informations strictement nécessaires. Les informations des nouveaux clients sont contenues dans un autre dataset, on considère que ces informations préremplies ont été communiquées au préalable via un questionnaire ou une enquête.
Enfin, le déploiement sur le web des deux applications doit se faire dans un cadre permettant une intégration et amélioration continues.

### Analyse exploratoire et feature engineering
L'analyse exploratoire permet déjà de faire apparaître des indicateurs plus ou moins importants, lorsque l'on regarde les corrélations avec les défauts de paiement notamment.
Etapes du nettoyage et du feature engineering (voir scripts : https://www.kaggle.com/willkoehrsen):
- **suppression des outliers** 
- **LabelEncoding** des features catégorielles dans le cadre de la préparation à la modélisation
- **observation des corrélations** après transformation

### Modélisation et MLFlow
Mise en place d'un processus de MLFlow pour **enregistrer les expériences** (modèles avec éventuels pipelines de transformations supplémentaires, paramètres et hyperparamètres, temps d'entraînement et de validation, différentes métriques de validations : AUC, Accuracy etc.) au travers d'une fonction unique. Dans le cas présenté dans le Notebook, la mise en place du MLFlow est intervenue postérieurement à l'exploration et l'optimisation des modèles ; par conséquent, toutes les expériences n'ont pas été enregistrées : une seule par modèle.
Choix de plusieurs **modèles de classification** :  
- **DummyClassifier** pour référence
- **LightGBM Classifier**
Pour ce dernier, deux techniques de **gestion du déséquilibre des données** ont été testées:
- les techniques de **"class_weight" et undersampling**, (intégrées dans les paramètres des modèles pour la 1ère technique), permettant de gérer le déséquilibre.

De même, pour chaque approche, **l'optimisation du score AUC** est d'abord recherchée, puis **un nouveau score à minimiser est créé**, à partir des résultats obtenus pour chaque classe avec la **fonction de coût = ((100 x FP) - (10 x TN) + 1 x (FN+FP+TN+TP)) / (FN+FP+TN+TP)**.
Enfin, dans chaque cas, un calcul des **prédictions sous forme de probabilité** est réalisé afin de rechercher **le meilleur seuil pour minimiser cette même fonction de coût**.
Au regard de l'ensemble des métriques, y compris les temps, et en comparant les ROC curves, le meilleur modèle est retenu : le modèle LightGBM avec paramétrage "class_weight" (hyperparamètres dans le Notebook et dans les mlruns).
Un pipeline de transformation des features et de modélisation est recréé afin d'être enregistré avec Joblib.

### Explicabilité globale et locale
Pour **l'explicabilité globale et locale, les Shap values des features** ont été utilisées.

### Analyse du Data Drift
Avec la librairie **Evidently**, le drift peut être mesuré pour **évaluer la pertinence du modèle dans le temps**. Cette analyse peut être effectuée à chaque fois qu'on enregistre un certain nombre de nouveaux clients (nombre ou période à déterminer). Ici on effectue l'analyse sur les jeux de données restreints (test_samp : 488 individus vs train_samp : 3076 individus).  
Le data drift a été effectué sur 347 colonnes. Plus de 35% des colonnes présentes du data drift, notamment les colonnes liées à l'âge des clients ("DAYS_BIRTH"). Cela impacte de nombreuses variables qui sont générées à partir de cette feature : APP_DAYS_EMPLOYED_DAYS_BIRTH_diff ou APP_SCORE1_TO_BIRTH_RATIO par exemples. Bien qu'il soit difficile d'y remédié dans notre cas, il s'agit d'une problématique dont on doit tenir compte dans l'entrainement de modèles de ML.

### Description du dossier API
Le choix a été fait de construire une API avec **FastAPI**.  
En plus du dossier (fichier Yaml) de workflows, décrivant les différentes tâches à effectuer, plusieurs fichiers sont indispensables :
- **le script python de l'API noté 'main.py'** détaillé plus bas
- **le script python du test unitaire (Pytest) noté 'test_main.py'** : ce dernier test lors du déploiement continue que l'API appelle bien le modèle pour effectuer des prédictions de classement (un individu en-dessous et un au-dessus sont testés)
- **les deux datasets réduits (dans un dossier 'ressources')** contenant respectivement les informations des clients références et celles des nouveaux clients. Ces datasets sont amenés à évoluer avec le temps et ne seront modifiés qu'ici.
- **le fichier 'model.joblib' (dans 'ressources') dans lequel le modèle entraîné est enregistré, ainsi que le pipeline de transformation des données**. Si le modèle change ou le pipeline changent, le fichier sera changé.
- **le fichier requirements.txt contenant les librairies nécessaires** à contruire l'environnement dans lequel l'API peut fonctionner (à modifier en cas d'utilisation de nouveaux modèles de références par exemple).
- **/!\ A noter qu'un script de lancement de l'API sous Azure est nécessaire (Configuration/General Settings/Startup command)** : 
> apt update
> apt-get install -y libgomp1
> gunicorn -w 2 -k uvicorn.workers.UvicornWorker main:app

A noter que l'API doit être installée sous une **Web App Azure Linux en Python 3.11**.
L'URL de l'application est la suivante : **https://basicwebappvl.azurewebsites.net**. Bien entendu, pour qu'elle fonctionne, elle nécessite d'être activée!

Concernant le script de l'API, **7 "routes" sont mises en place** pour interagir avec l'utilisateur de l'application (interface du conseiller clientèle).
**Différentes méthodes GET** pour obtenir :
- la liste des ids des nouveaux clients pour vérifier que le client est bien enregistré
- la liste des features qui sont utilisées pour la modélisation
- envoyer le numéro du client sélectionné et recevoir ses informations (format json)
- envoyer le json d'un client et recevoir sa prédiction de classement
- envoyer le numéro d'un client et recevoir son explicabilité locale (Shap values)
- envoyer le nom d'une feature et recevoir les valueurs de cette features pour les individus classés 0 et ceux classés 1


### Description du dossier Streamlit
L'interface utilisateur est une application Streamlit.  
En plus du dossier (fichier Yaml) de workflows, décrivant les différentes tâches à effectuer, 3 fichiers seulement sont nécessaires :
- **le script de l'application nommé 'dashboard.py'** détaillé plus bas
- **les fichiers png du logo de la société fictive et celui d'OCR**
- **le fichier des requirements** limités à l'application
- **/!\ A noter qu'un script de lancement de l'API sous Azure est nécessaire (Configuration/General Settings/Startup command)** : 
> python -m streamlit run dashboard.py --server.port 8000 --server.address 0.0.0.0

A noter que le dashboard doit être installé sous une **Web App Azure Linux en Python 3.11**.
L'URL de l'application est la suivante : **https://basicwebappvl.azurewebsites.net**. Bien entendu, pour qu'elle fonctionne, elle nécessite d'être activée AINSI QUE l'API!

Description de l'expérience utilisateur pour un client :
- le conseiller est invité à sélectionner le numéro d'un client. Un score de prédiction d'appartenance à la classe 0 (solvable) est ainsi affiché (sous forme de gauge plot, par rapport au seuil déterminé de 60%). POur ledit client, il est possible de sétectionner une feature choisie et de voir qu'elle est la valeur de cette valeur sur un graphique de 'density plots' des 2 groupes de population (classes 0 et 1). De même, les résultats des shap values (interprétabilité locale) nous renseigne sur les 10 variables qui 'tirent' le plus l'individu en question vers le groupe des 0 ou celui des 1 (non solvables). 

### Tests et workflows
La plateforme d'hébergement choisie est **Azure Web App**.  
Afin de mettre en place un **processus d'intégration/amélioration continues**, le code est hébergé sur des **repo Git distants** et le déploiement réalisé par les **actions Github** communiquant avec l'hébergeur. De cette manière, des modifications peuvent être réalisées puis contrôlées d'abord dans un **environnement virtuel local** défini, puis éventuellement déployées dans une **nouvelle branche** avant d'être envoyées à la branche principale.  
Il a été décidé de séparer complètement le déploiement de l'API de celui de l'application et des projets Github distincts ont été créés :
- pour l'API : https://github.com/DDesou/Projet7_VL
- pour l'interface utilisateur : https://github.com/DDesou/Projet7_Streamlit

L'API est déployée (ou modifiée) après que les tests unitaires aient été validés (**tests Pytest** intégrés dans le déploiement). Les scripts des test effectués sont contenus dans le fichier test_main.py.  
L'application Streamlit est déployée (ou modifiée) après s'être assuré que toutes les modifications ont été d'abord enregistrées, afin d'éviter les bugs éventuels au moment du changement de version.
