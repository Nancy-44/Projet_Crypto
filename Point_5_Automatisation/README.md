
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
Il récupère les données JSON de l'API (/historical, /predict et /latest) et affiche un dashboard utilisateur.
