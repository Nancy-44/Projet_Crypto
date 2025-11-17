# Logins pour se sonnecter à l'API : 
API_USER=admin  
API_PASSWORD=secret123  

# Démarche pour se connecter à l'API avec un docker compose
# 1. Lancer tous les services avec docker compose
docker compose up -d # lance en arrière plan  
docker compose ps # vérifier que les containers sont sains et up  

# 2. Vérifier données dans MongoDB
docker compose exec mongo bash # pour se connecter au container mongoDB  
mongosh -u <user> -p <password> - authenticationDatabase admin # pour lancer shell Mongo  
> use crypto  
> db.klines.find().limit(5)  

si mongoDB vide, laisser tourner service streaming via docker compose  

# 3. Vérifier données dans PostgreSQL
docker compose exec postgres bash # pour se connecter au container postgreSQL  
psql -U <user> -d <db> # lancer psql  
> \dt # pour lister les tables  
> SELECT * FROM klines LIMIT 5; # vérifier les données  
 si postgreSQL vide, relancer le service etl via docker compose  

# 4. Tester les endpoints de l'API
## A. health
curl -u <user>:<password> "http://localhost:8000/health"
## B. Historical (via PostgreSQL)
curl -u <user>:<password> "http://localhost:8000/historical?symbol=BTCUSDT&limit=5"
## C. Latest (via MongoDB)
curl -u <user>:<password> "http://localhost:8000/latest?symbol=BTCUSDT"
## D. Predict (ML)
curl -u admin:admin123 -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{"symbol": "BNBUSDT", "interval": "15m"}'

# 5. Si Message d'erreur, vérifier les logs
docker compose logs -f api  
docker compose logs -f streaming
