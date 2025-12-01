# Projet_CryptoBot
De nos jours, le monde des crypto commence à prendre une place importante et grossi.  
Il s’agit de marchés financiers très volatiles et instables se basant sur la technologie de la Blockchain.  
Le but principal de ce projet est de créer un bot de trading basé sur un modèle de Machine Learning et qui investira sur des marchés crypto.  

Ce projet est entièrement automatisé, monitoré et reproductible, avec une architecture composée des services suivants :  

1. API : interface FastAPI pour récupérer les prédictions et les données historiques et en temps réel.  
2. Airflow : orchestration du pipeline ETL et streaming.  
3. GitLab : CI/CD avec runner Docker.  
4. Grafana & Prometheus : monitoring des métriques systèmes, de l’API et des conteneurs Docker via cAdvisor.  
5. Streamlit : dashboard utilisateur pour visualiser les données et les prédictions.  

# Lancement du CryptoBot  
Le projet peut être lancé facilement via Docker Compose à la racine du dépôt GitHub  
> docker-compose up --build -d  
Cela démarre tous les services nécessaires (API, Airflow, Streamlit, monitoring).  

# 1. API   
> url: http://localhost:8000/docs  
> Identifiant : admin  
> Mot de passe : secret123      
Endpoints principaux : /predict, /historical, /latest.  

# 2. Airflow  
## Construire l'image d'Airflow  
> cd Projet_Crypto/Etape_5/airflow  
> docker build -t crypto_airflow:latest .  
## Lancer Airflow  
> docker-compose up --build -d  

## Nettoyage avant de relancer  
> docker-compose down -v  
> docker system prune -a --volumes -f  

## Consulter les logs en cas d'erreur  
> docker-compose logs -f api  
> docker-compose logs -f airflow-webserver  
> docker-compose logs -f airflow-scheduler  

# 3. Gitlab  
## Installer Gitlab runner  
> curl -L http://packages.gitlab.com/install/repositories/runner/gitlab-runner/script.deb.sh | sudo bash  
> sudo apt-get install gitlab-runner -y  
> gitlab-runner --version # vérification installation  
## Enregistrer GitLab runner avec 'Projet_crypto"  
> sudo gitlab-runner register # renseigner gitLab URL, token, name runner(runner_crypto), executor(docker), defaut docker image (python:3.11)  
## Configurer GitLab runner pour utiliser Docker  
> sudo usermod -aG docker gitlab-runner  
> sudo systemctl restart gitlab-runner  
Vérifier sous GitLab que runner est "online" (vert)  
Vérifier que les fichiers .gitlab-ci.yml et docker-compose.yml sont à la racine du 'Projet_Crypto/Etape_5'   
## Initialiser dépôt Git local   
> cd ~/Projet_Crypto/Etape_5  
> git init # initialiser Git  
> git remote add origin git@gitlab.com:nancy44/projet_crypto.git  
## Ajouter les fichiers  
> git add . # Ajouter tous les fichiers  
> git commit -m "Initial commit et pipeline CI/CD."  
> git branch -M main # se placer sur la branche main  
> git push -u origin main # renseigner username GitLab et password GitLab

# 4. Prometheus & Grafana  
> Prometheus : http://<IP_VM>:9000/  
Vérifier metrics cAdvisor (Prometheus => Status => Target_health si cAdvisor up (ok)  
> Grafana : http://<IP_VM>:3000  
> Identifiant : admin  
> Mot de passe : admin # Importer le fichier.json pour le dashboard  
Visualisation des metrics suivantes :  
CPU Usage  
Memory Usage
API Requests & Latency  
PostgreSQL Connections  
Conteneurs Docker actifs (via cAdvisor)  

# 5. Streamlit  
> url: http://localhost:8501/  
Permet de visualiser les prédictions, la dernière valeur temps réel et l’historique sous forme de tableau et de graphiques.  





### Equipe Projet :  
Nancy Frémont  ([GitHub](https://github.com/) / [LinkedIn](http://linkedin.com/))  
Philippe Kirstetter-Fender  ([GitHub](https://github.com/) / [LinkedIn](http://linkedin.com/))  
Florent Rigal  ([GitHub](https://github.com/) / [LinkedIn](http://linkedin.com/))  
Thomas Saliou  ([GitHub](https://github.com/7omate) / [LinkedIn](http://linkedin.com/))  

### Encadrant du projet :  
Rémy Dallavalle    


### Sources  
#### Enoncé du Projet : [Google Doc](https://docs.google.com/document/d/1kD6haSp3reTUA8sMsd0x9z6FpJ7rfcZydrmZJOi40Ak/edit?tab=t.0)  

#### Rapports  
Voici ci-dessous les liens vers les rapports remis à chaque étape du projet:  
A. Récolte des données : [Fichier descriptif du traitement appliqué_point1_Projet_Crypto.pdf](https://github.com/user-attachments/files/23800266/Fichier.descriptif.du.traitement.applique_point1_Projet_Crypto.pdf)  
B. Stockage de la donnée : [Rapport_Projet_CryptoBot_Partie2.pdf](https://github.com/user-attachments/files/23800333/Rapport_Projet_CryptoBot_Partie2.pdf)  
C. Consommation de la donnée (Machine Learning) : [Rapport_Projet_CryptoBot_Partie 3_finalise.pdf](https://github.com/user-attachments/files/23800358/Rapport_Projet_CryptoBot_Partie.3_finalise.pdf)  
D. Mise en production (API) : [Rapport_Projet_CryptoBot_Partie 4.pdf](https://github.com/user-attachments/files/23800416/Rapport_Projet_CryptoBot_Partie.4.pdf)  
E. Automatisation des flux et Monitoring : [Rapport_Projet_CryptoBot_Partie_5.pdf](https://github.com/user-attachments/files/23800449/Rapport_Projet_CryptoBot_Partie_5.pdf)  


#### Présentation : [Presentation](https://docs.google.com/presentation/d/1H5-MovAMEX6wPQzWxFyq7Xm6sqe9KCvi9rVl7dXg2G0)  

Ce projet a été mené dans le cadre de la formation Data Engineer réalisée chez Datascientest.
