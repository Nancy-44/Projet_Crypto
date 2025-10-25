## Partie 2 : Stockage de la donnée
Il s’agit de choisir la solution de stockage la plus adaptée.
Choix de ou des bases de données

# Critères sur la nature et l’usage des données : 
- les données récoltées sont structurées (timestamp, OHLCV).
- les données arrivent soit en flux continu (streaming), soit en bloc (batch via API).
- les données historiques serviront à entraîner des modèles de Machine Learning et à la visualisation.
- les données en streaming serviront à valider le modèle de Machine Learning et pour des requêtes analytiques (décision d’achat ou de vente) en temps réel.

# Avantages et Inconvénients des bases de données proposées : 
SYSTEME	TYPE	AVANTAGES	INCONVENIENTS
SQL (MySQL, PostgreSQL)	Relationnelle	- simple, structuré
- bon pour les jointures
- connu et bien documenté	- moins efficace pour les données massives en temps réel
- moins flexible pour schémas évolutifs
ElasticSearch	NoSQL (search et analyse en temps réel)	- indexation rapide
- agrégation sur séries temporelles
- très bon outil de visualisation (Kibana)	- moins adapté pour relations complexes
- pas optimal pour le stockage à long terme
MongoDB	NoSQL (document)	- flexible
- bon pour le streaming 
- stockage Json
- scalabilité	- moins adapté aux jointures complexes
- pas très performant pour les grosses requêtes analytiques
Snowflake	Data Warehouse Cloud	- bon pour les gros volumes analytiques
- langage SQL
- optimisé pour la BI et la Data Science	- coût élevé
- période d’essai trop courte
- moins bon pour ingestion en temps réel
BigQuery	Data Warehouse Cloud	- très scalable
- bon pour les requêtes sur de gros volumes
- facile à intégrer avec GCP AI tools	- latence avec données en batch
- non créé pour l’ingestion en temps réel





En comparant les critères des données avec les avantages/inconvénients des bases de données, il ressort : 
- les données historiques peuvent être stockées avec Snowflake ou PostgreSQL.
Le principal inconvénient pour Snowflake est son coût élevé et sa courte période d’essai qui ne couvre pas la durée totale de la formation.
- les données en streaming peuvent être stockées avec ElasticSearch ou MongoDB. 
Pour les données websockets, je pense que vous pouvez exclure Elasticsearch, vu qu'il est surtout utile pour le traitement de données textuel et d'indexation. 

Ainsi, au vue des explications ci-dessus, il a été retenu postgreSQL pour les données historiques et MongoDB ElasticSearch pour les données en streaming.

Vous pouvez très bien avoir 2 sgbd : un relationnel, pour les données historisées, et un no-sql pour les données temps réel. Ce "double" traitement se rapproche donc d'une architecture Lambda (à détailler…)







 
## Diagramme UML (selon convention d’écriture française)

Constats : 
- Un symbole (Marché) a plusieurs Klines (Chandeliers). Chaque Kline appartient à un seul symbole.
- Un symbole peut avoir plusieurs intervalles. Un intervalle peut être lié à plusieurs symboles.
- Chaque kline (une Kline est définie par un triplet : un symbole, un intervalle, un timestamp d’ouverture) n’appartient qu’à un seul intervalle. Un intervalle peut être lié à plusieurs klines.

Pour passer en diagramme UML, je me suis basée sur les cardinalités maximales uniquement, donc cela correspond à une relation (N, 1) entre symbol et Klines ; puis une relation (N, N) entre symbol et interval.
Selon la 1ère règle de Merise : Si relation (1,N) ou (N,1), la clé primaire côté N descend dans l’entité côté 1 et devient clé étrangère dans la table Klines.
Selon la 2ème règle de Merise : si cardinalité (N,N), l’association se transforme en table INTERVAL supplémentaire. Clés primaires respectives de chaques entités descendent dans table INTERVAL créée. 

 <img width="873" height="557" alt="image" src="https://github.com/user-attachments/assets/98a1b4f7-cde7-42fd-bf2a-5535913a7b27" />


Symbol_id, interval_id et klines_id : primary keys. Il s’agit d’index des tables.
Symbol_name : BTCUSDT, ETHUSDT, etc.
Base_asset : BTC
Quote_asset : USDT
Interval_name : “1m”, “5m”, “15m”, “1h”, “4h”, “1d”
Seconds : durée en seconde
Open_time : heure ouverture du klines
Close_time : heure de fermeture du klines
Timestamps : date de création

Pas de champs calculé dans les tables.


# Intervalles sur Binance (page Market data endpoints via developers.binance)
Intervalle Binance	Durée réelle	Utilité
1m			1 minute	Micro-analyse, scalping, algo-trading haute fréquence
5m			5 minutes	Analyse courte durée, alerting
15m			15 minutes	Observation intra-journalière
1h			1 heure		Moyen terme, journalière
4h			4 heures	Suivi de tendance
1d			1 jour		Analyse long terme, indicateurs journaliers

Plusieurs types de trading : scalping non utile pour notre projet. On s’intéressera plutôt au moyen, long terme=> retrait des 1m et 5m. Voir pour rajouter 1w

Pour le projet crypto, nous voulons entraîner un modèle de Machine Learning, puis calculer des indicateurs techniques  MSE, MAO, RMSE).
Pour le machine Learning, il nous faut beaucoup de données (précision fine) donc un intervalle entre 1h et 4h.
Pour les indicateurs, besoin de moins de données donc entre 15 minutes et 1h.
Pour la visualisation, on peut passer à un intervalle 1h, 1d ou 1w.
On peut donc garder la liste d’intervalles suivantes : [15min, 1h, 4h, 1d, 1w].


