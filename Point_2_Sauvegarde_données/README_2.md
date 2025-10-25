## Partie 2 : ETL
Il s’agit de choisir la solution de stockage la plus adaptée.  
# 1. Choix de ou des bases de données
Critères sur la nature et l’usage des données :  
- les données récoltées sont structurées (timestamp, OHLCV).  
- les données arrivent soit en flux continu (streaming), soit en bloc (batch via API).  
- les données historiques serviront à entraîner des modèles de Machine Learning et à la visualisation.  
- les données en streaming serviront à valider le modèle de Machine Learning et pour des requêtes analytiques (décision d’achat ou de vente) en temps réel.  

# Avantages et Inconvénients des bases de données proposées : 
<img width="804" height="617" alt="image" src="https://github.com/user-attachments/assets/ff6b2ce0-3de7-41cf-8865-e576c8d25257" />






En comparant les critères des données avec les avantages/inconvénients des bases de données, il ressort :  
- les données historiques peuvent être stockées avec Snowflake ou PostgreSQL.  
Le principal inconvénient pour Snowflake est son coût élevé et sa courte période d’essai qui ne couvre pas la durée totale de la formation.  
- les données en streaming peuvent être stockées avec MongoDB. ElasticSearch n’est pas adapté ici car cet outil est surtout utile pour le traitement de données textuel et d’indexation.  
=> Ainsi, au vue des explications ci-dessus, il a été retenu postgreSQL pour les données historiques et MongoDB pour les données en streaming.  
Ce traitement double, à savoir une base de données relationnelles pour les données historisées et une base de données no-sql pour les données en temps réel, s’appelle une architecture Lambda.  


# 2. Diagramme UML (selon convention d’écriture française)

Constats :  
- Un symbole a plusieurs Klines (Chandeliers). Chaque Kline appartient à un seul symbole.  
- Un symbole peut avoir plusieurs intervalles. Un intervalle peut être lié à plusieurs symboles.  
- Chaque kline (une Kline est définie par un triplet : un symbole, un intervalle, un timestamp d’ouverture) n’appartient qu’à un seul intervalle. Un intervalle peut être lié à plusieurs klines.

Pour passer en diagramme UML, je me suis basée sur les cardinalités maximales uniquement, donc cela correspond à une relation (N, 1) ou (1,N).  
Selon la 1ère règle de Merise : Si relation (1,N) ou (N,1), la clé primaire côté N descend dans l’entité côté 1 et devient clé étrangère dans la table Klines. 

 <img width="883" height="659" alt="image" src="https://github.com/user-attachments/assets/7b5f5657-478d-496b-ae0c-2d4f817d171e" />

  
Symbol_id, interval_id et klines_id : primary keys. Il s’agit d’index des tables.  
Symbol_name : BTCUSDT, ETHUSDT, etc.  
Base_asset : BTC  
Quote_asset : USDT  
Interval_name : “15m”, “1h”, “4h”, “1d”, “1w”  
Seconds : durée en seconde 
Open_time : heure ouverture du klines  
Close_time : heure de fermeture du klines  
Timestamps : date de création  

Pas de champs calculé dans les tables.  

# 3. Justification du choix des intervalles  
Intervalles sur Binance (page Market data endpoints via developers.binance)  
Type de trading	Intervalle Binance recommandé	Pourquoi / Utilisation  
Scalping	1m, 3m, 5m	Très court terme, prise de positions sur quelques minutes, nécessite réactivité.  
Day trading	15m, 30m, 1h	Permet de suivre les tendances intrajournalières sans le bruit des 1m/5m.  
Swing trading	4h, 6h, 12h	Capture les mouvements sur plusieurs jours, moins sensible aux fluctuations mineures.  
Position trading / Long terme	1D, 3D, 1W	Analyse macro, suit les tendances sur plusieurs semaines ou mois.  

<img width="945" height="479" alt="image" src="https://github.com/user-attachments/assets/adc7ac25-901e-4efe-818f-ccde549d70f8" />

D’après le tableau ci-dessus, plusieurs types de trading existent : scalping, day trading,  swing trading et analyse long terme.  
Pour le projet crypto, nous voulons entraîner un modèle de Machine Learning, puis calculer des indicateurs techniques (MSE, MAO, RMSE). Ainsi, nous n’avons pas besoin de scalping.
Pour le machine Learning, il nous faut beaucoup de données (précision fine) donc un intervalle entre 1h et 4h.
Pour les indicateurs, nous avons besoin de moins de données donc entre 15 minutes et 1h.
Pour la visualisation, on peut passer à un intervalle 1d ou 1w.  
On peut donc garder la liste d’intervalles suivantes : [15min, 1h, 4h, 1d, 1w].
 
Données historiques sauvegardées dans PostgreSQL
 
Nombre de datas par symbol : environ 30000.  
 
