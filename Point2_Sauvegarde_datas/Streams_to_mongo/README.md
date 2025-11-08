# Lancer le script

## Pour obtenir des données en streaming

    # docker compose up 
    #pip install pymongo python-binance 
    # install spark
    python ../1_collecte_datas/1-streams.py

## Pour obtenir des données historiques

    # docker compose up 
    #pip install pymongo python-binance 
    python klines2mongo.py

# Voir les données

Les données sont collectées avec une collection par symbol.

    docker exec -it my_mongo bash
    mongosh -u user -p password
    use binance_klines
    show collections # une collection par symbol
    db.BTCUSDT.findOne()


## exemple de données
```
{
  _id: ObjectId('68f0f0fbb58dfa2bbac90d8d'),
  open_time: ISODate('2025-09-25T18:00:00.000Z'),
  open: 108833.63,
  high: 109801.78,
  low: 108631.51,
  close: 109760.2,
  volume: 1427.42291,
  close_time: ISODate('2025-09-25T18:59:59.999Z'),
  symbol: 'BTCUSDT'
}
```

# Améliorations

On voudrait ecouter les données en streaming et a partir de ces données consolider la base de données pour obtenir les données a la minute, l'heure, 24h. Cela ne semble pas trivial.

# Visualization
Un outil existe qui s'appelle Mongodb _Charts_ mais il ne semble plus possible a l'heure actuelle de le déployer localement:
```
Docker Image Format v1 and Docker Image manifest version 2, schema 1 support has been removed. Suggest the author of quay.io/mongodb/charts:latest to upgrade the image to the OCI Format or Docker Image manifest v2, schema 2. More information at https://docs.docker.com/go/deprecated-image-specs/
```
Voir le script dans notebooks, ViewMongo.ipynb
