# Structure du projet données historiques Binance

<img width="502" height="423" alt="image" src="https://github.com/user-attachments/assets/a417968c-583a-4b01-8a45-339106e68051" />



Créer fichier docker_compose.yaml avec les identifiants et mots de passe postgres et pgadmin4

# Etape pour construire le pipeline etl
docker-compose build --no-cache # pour construire les images docker  
docker-compose up -d postgres pgadmin # pour lancer postgres et pgadmin  
pip install pandas requests sqalchemy psycopg2-binary  
docker-compose run --rm etl python3 extract.py # pour lancer extraction  
docker-compose run --rm etl python3 transform.py # pour lancer transformations  
docker-compose run --rm etl python3 load.py # pour charger les données dans postgres  


# Accéder à PgAdmin4
depuis navigateur Web : http://IP_VM:8080  
Renseigner son email et son password  
Sur PgAdmin4 : faire add_new_serveur et renseigner name, host, port, username et mot_de_passe.

Visualisation des données :    
<img width="559" height="262" alt="image" src="https://github.com/user-attachments/assets/31bcffda-0f23-4a11-a1b8-b987ad6fad1c" />
<img width="1039" height="208" alt="image" src="https://github.com/user-attachments/assets/04200721-627f-4021-b20d-2cd203440206" />
<img width="778" height="192" alt="image" src="https://github.com/user-attachments/assets/37a77800-d527-4e90-9313-133b40bf176e" />


Les données sont collectées (environ 30000 datas par symbol) 
sur 4 symboles : BNB, BTC, ETH, SOL et plusieurs intervalles [15m, 1h, 4h, 1d, 1w].

# Arrêter les services
docker-compose down  
docker-compose down -v # pour supprimer les volumes



