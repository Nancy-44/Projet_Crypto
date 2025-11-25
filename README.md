# Projet_CryptoBot
De nos jours, le monde des crypto commence à prendre une place importante et grossi.  
Il s’agit tout simplement de marchés financiers très volatiles et instables se basant sur la technologie de la Blockchain.  
Le but principal de ce projet est de créer un bot de trading basé sur un modèle de Machine Learning et qui investira sur des marchés crypto.  

Nous avons créé une arborescence comprenant les services suivants : airflow, api, machine learning (ml), streaming (récupération des données via Websocket binance), etl (pipeline de récupération des données historiques via API Binance), monitoring (via Prometheus et Grafana) et streamlit (accès à dashboard utilisateur).
Notre projet est entièrement automatisé et monitoré garantissant sa reproductibilité, sa scalabilité et sa fiabilité.  

L'accès à l'API, à Prometheus&Grafana et à Streamlit est détaillée ci-dessous. Il suffit pour cela de lancer le docker-compose général pour pouvoir accéder à chacun des services.  

# Airflow  
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

# Gitlab  
## Installer Gitlab runner  
> curl -L http://packages.gitlab.com/install/repositories/runner/gitlab-runner/script.deb.sh | sudo bash  
> sudo apt-get install gitlab-runner -y  
> gitlab-runner --version # vérification installation  
## Enregistrer GitLab runner avec 'Projet_crypto"  
> sudo gitlab-runner register # renseigner gitLab URL, token, name runner(runner_crypto), executor(docker), defaut docker image (python:3.11)  
## Configurer GitLab runner pour utiliser Docker  
> sudo usermod -aG docker gitlab-runner  
> sudo systemctl restart gitlab-runner  
## Vérifier sous GitLab que runner est "online" (vert)  
## Ajouter fichiers .gitlab-ci.yml et docker-compose.yml à la racine du 'Projet_Crypto/Etape_5'  
## Initialiser dépôt Git local   
> cd ~/Projet_Crypto/Etape_5  
> git init # initialiser Git  
> git remote add origin git@gitlab.com:nancy44/projet_crypto.git  
## Ajouter les fichiers  
> git add . # Ajouter tous les fichiers  
> git commit -m "Initial commit et pipeline CI/CD."  
> git branch -M main # se placer sur la branche main  
> git push -u origin main # renseigner username GitLab et password GitLab

# API  
> se connecter via http://localhost:8000/docs (identifiant : admin, mot de passe : admin123).  

# Prometheus & Grafana  
> Prometheus : http://<IP_VM>:9000/ # Vérifier metrics cAdvisor (Prometheus => Status => Target_health si cAdvisor up (ok)  
> Grafana : http://<IP_VM>:3000 (login : admin, password : admin) # Importer le datasource.yml pour le dashboard  
Visualisation des metrics suivantes :  
CPU Usage  
Memory Usage  
Network Receive  
Network Transmit

# Streamlit  
> Accès via http://localhost:8501/  
 



### Groupe Projet :  
Nancy Frémont  ([GitHub](https://github.com/) / [LinkedIn](http://linkedin.com/))  
Philippe Kirstetter-Fender  ([GitHub](https://github.com/) / [LinkedIn](http://linkedin.com/))  
Florent Rigal  ([GitHub](https://github.com/) / [LinkedIn](http://linkedin.com/))  
Thomas Saliou  ([GitHub](https://github.com/7omate) / [LinkedIn](http://linkedin.com/))  

### Encadrant du projet :  
Rémy Dallavale  


### Source document  
[Google Doc](https://docs.google.com/document/d/1kD6haSp3reTUA8sMsd0x9z6FpJ7rfcZydrmZJOi40Ak/edit?tab=t.0)  

Ce projet a été mené dans le cadre de la formation Data Engineer réalisée chez Datascientest.  
