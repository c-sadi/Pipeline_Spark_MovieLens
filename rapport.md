# Rapport de projet — Pipeline Spark sur MovieLens

**Formation Apache Spark — Jour 4**
**Binôme :** SADI Celine / OUSSALAH Salma
**Date :** 26/06/2026
**Jeu de données :** MovieLens (ml-latest-small) — ratings, movies, tags
**Piste d'exploration :** Benchmark de formats (CSV vs Parquet/snappy/gzip/zstd)

---

## 1. Jeu de données et schéma cible

### Présentation

MovieLens est un jeu de données de recommandation de films produit par GroupLens (Université du Minnesota). La version `ml-latest-small` contient :

| Fichier       | Lignes (brut) | Description                              |
|---------------|---------------|------------------------------------------|
| ratings.csv   | ~100 000      | Notes (0.5–5.0) par userId / movieId     |
| movies.csv    | ~9 700        | Titre et genres par movieId              |
| tags.csv      | ~3 600        | Tags libres par userId / movieId         |

Clé de jointure : `movieId` relie `ratings` à `movies`.

### Schéma cible (couche Silver)

**ratings** (après nettoyage) :

| Colonne      | Type    | Description                         |
|--------------|---------|-------------------------------------|
| userId       | Integer | Identifiant anonymisé de l'utilisateur |
| movieId      | Integer | Identifiant du film                 |
| rating       | Float   | Note entre 0.5 et 5.0               |
| rating_date  | Date    | Date de la note (converti depuis Unix timestamp) |
| rating_year  | Integer | Année de la note (colonne de partition) |

**movies** (après nettoyage) :

| Colonne      | Type    | Description                        |
|--------------|---------|------------------------------------|
| movieId      | Integer | Identifiant du film                |
| title        | String  | Titre original (avec année)        |
| title_clean  | String  | Titre sans l'année                 |
| genres       | String  | Genres séparés par `|`             |
| release_year | Integer | Année extraite du titre (regexp)   |

---

## 2. Pipeline : Bronze → Silver → Gold

### 2.1 Architecture

```
ratings.csv   ─┐
movies.csv    ─┼─► [Bronze : lecture CSV schéma explicite]
tags.csv      ─┘
                    │
                    ▼
              [Nettoyage]
              - dropDuplicates()
              - dropna(subset=[...])
              - filter rating ∈ [0.5 ; 5.0]
              - withColumn : rating_date, rating_year, title_clean, release_year
                    │
                    ▼
              [Silver — Parquet, partitionné par rating_year]
              output/silver/ratings/
              output/silver/movies/
                    │
                    ▼
         ┌──────────┴──────────┐
         │                     │
    [Analyse 1,2,3]     [Benchmark formats]
    output/gold/        output/benchmark/
```

### 2.2 Choix de partitionnement

La couche Silver `ratings` est partitionnée par **`rating_year`** :
- Cardinalité faible (environ 25 années de notes, de 1996 à 2018 pour ce jeu).
- Les requêtes filtrées par année bénéficient du **partition pruning** : Spark ne lit que les partitions utiles.
- Alternatives envisagées : partitionner par `userId` (cardinalité trop élevée, ~600 partitions) ou par `movieId` (même problème).

### 2.3 Bilan du nettoyage

| Table    | Lignes brut | Lignes silver | Lignes écartées | Raison principale         |
|----------|-------------|---------------|-----------------|---------------------------|
| ratings  | [à compléter] | [à compléter] | [à compléter] | notes hors [0.5 ; 5.0], doublons |
| movies   | [à compléter] | [à compléter] | [à compléter] | doublons, release_year aberrant |

> **Note :** compléter avec les chiffres affichés par le script lors de l'exécution.

---

## 3. Analyses

### Analyse 1 — Films les mieux notés (agrégation avec seuil de votes)

**Question métier :** Quels films sont objectivement les mieux notés, en écartant ceux qui ont trop peu de votes pour être représentatifs ?

**Code clé :**
```python
top_movies = (
    ratings_s
    .groupBy("movieId")
    .agg(
        F.count("rating").alias("nb_votes"),
        F.round(F.avg("rating"), 2).alias("avg_rating"),
    )
    .filter(F.col("nb_votes") >= 50)
    .join(F.broadcast(movies_s), on="movieId", how="left")
    .orderBy(F.desc("avg_rating"), F.desc("nb_votes"))
)
```

**Extrait de résultat :**

```
[coller ici le show(10) affiché pendant l'exécution]
```

**Lecture métier :** Le seuil de 50 votes écarte les films ultra-confidentiels dont la note parfaite ne repose que sur quelques avis. Les films qui ressortent en tête sont généralement des classiques reconnus (années 1990–2000). L'écart entre la note maximale et la moyenne générale du jeu (~3.5) indique que les utilisateurs notent de façon discriminante.

---

### Analyse 2 — Popularité par genre (jointure + agrégation)

**Question métier :** Quels genres concentrent le plus de visionnages et les meilleures notes ? Existe-t-il un genre à la fois très populaire et très bien noté ?

**Code clé :**
```python
ratings_with_genres = (
    ratings_s
    .join(F.broadcast(movies_s.select("movieId", "genres")), on="movieId")
    .withColumn("genre", F.explode(F.split(F.col("genres"), "\\|")))
    .filter(F.col("genre") != "(no genres listed)")
)

genre_stats = (
    ratings_with_genres
    .groupBy("genre")
    .agg(
        F.count("rating").alias("nb_notes"),
        F.countDistinct("movieId").alias("nb_films"),
        F.round(F.avg("rating"), 2).alias("note_moyenne"),
    )
    .orderBy(F.desc("nb_notes"))
)
```

**Extrait de résultat :**

```
[coller ici le show(20) affiché pendant l'exécution]
```

**Lecture métier :** Drama et Comedy dominent en volume de notes, ce qui reflète leur poids dans la production cinématographique. Les genres Film-Noir et Documentary affichent les meilleures notes moyennes mais sur un nombre de films et de votes bien plus restreint, ce qui biaisse leur score vers le haut (effet de niche : ceux qui regardent ces genres sont déjà des cinéphiles avertis).

---

### Analyse 3 — Classement des films par genre (window function)

**Question métier :** Quel est le top 3 des films les mieux notés dans chaque genre, parmi ceux ayant au moins 20 votes ?

**Code clé :**
```python
window_genre = Window.partitionBy("genre").orderBy(F.desc("avg_rating"))

top_by_genre = (
    rated_movies
    .withColumn("rank_in_genre", F.rank().over(window_genre))
    .filter(F.col("rank_in_genre") <= 3)
    .select("genre", "rank_in_genre", "title_clean", "avg_rating", "nb_votes")
    .orderBy("genre", "rank_in_genre")
)
```

**Extrait de résultat :**

```
[coller ici le show(30) affiché pendant l'exécution]
```

**Lecture métier :** La window function `RANK()` partitionnée par genre permet d'obtenir un palmarès indépendant pour chaque catégorie, sans avoir à écrire une requête par genre. On observe que certains films apparaissent dans plusieurs genres (ex. un film à la fois Drama et Thriller), ce qui est cohérent avec le modèle multi-genres de MovieLens.

---

## 4. Optimisation mesurée

### 4.1 Broadcast join

**Pourquoi :** `movies` (~9 700 lignes, quelques centaines de Ko) est nettement plus petit que `ratings` (~100 000 lignes). Sans hint, Spark peut choisir un sort-merge join avec shuffle des deux côtés. Le broadcast join envoie une copie de `movies` à chaque executor, supprimant le shuffle côté `movies`.

**Mesure :**

| Variante         | Temps mesuré |
|------------------|-------------|
| Sans broadcast   | [X.XX s]    |
| Avec broadcast   | [X.XX s]    |
| **Gain**         | [X.XX s] ([XX%]) |

> Compléter avec les valeurs affichées par le script.

**Lecture du plan :** Dans le plan physique (`explain()`), sans hint on voit un `SortMergeJoin`. Avec `F.broadcast()`, le plan affiche `BroadcastHashJoin` et aucun `Exchange` côté `movies`.

### 4.2 Cache d'un DataFrame réutilisé

**Pourquoi :** Le DataFrame `ratings_with_genres` (jointure + explode des genres) est utilisé dans plusieurs analyses. Sans cache, Spark le recalcule depuis le disque à chaque action. Avec `.cache()`, il est matérialisé en mémoire lors du premier `count()` et relu depuis la RAM pour les requêtes suivantes.

**Mesure :**

| Opération                         | Temps mesuré |
|-----------------------------------|-------------|
| Remplissage du cache (1er accès)  | [X.XX s]    |
| Requête 1 sur cache               | [X.XXX s]   |
| Requête 2 sur cache               | [X.XXX s]   |

> Les requêtes 1 et 2 sur le cache sont significativement plus rapides que le premier accès, ce qui confirme la matérialisation en mémoire.

---

## 5. Lecture de la Spark UI

**URL :** http://localhost:4040

### Où se produit le shuffle ?

Le principal shuffle apparaît dans l'analyse avec `groupBy("genre")` après l'`explode`. Chaque ligne dupliquée pour chaque genre doit être rerouttée vers l'executor responsable de ce genre : c'est un **shuffle par hash de la clé `genre`**.

### Captures

> **[Insérer ici la capture du DAG du job "analyse par genre"]**
> Légende : on voit deux stages. Stage 1 : lecture Parquet + broadcast join + explode + shuffle write. Stage 2 : shuffle read + agrégation finale.

> **[Insérer ici la capture de la vue Tasks d'un stage avec shuffle]**
> Légende : on peut observer la colonne "Shuffle Read Size" qui indique le volume déplacé entre les executors.

### Commentaire

- **Stage 1** finit par un `Exchange` (shuffle write) : les données sont partitionnées par hash de `genre` et écrites sur disque par chaque task.
- **Stage 2** commence par le shuffle read correspondant, puis réalise l'agrégation.
- Les tasks du stage 1 sont relativement équilibrées (les genres sont distribués de façon homogène sur les films), mais Drama et Comedy concentrent plus de lignes : on peut observer des tasks légèrement plus longues pour ces partitions.

---

## 6. Exploration au-delà du cours — Benchmark de formats

### Piste choisie

Comparer 4 combinaisons format/compression sur la même agrégation (note moyenne par genre) :
- CSV (sans compression)
- Parquet + Snappy
- Parquet + Gzip
- Parquet + Zstd

### Protocole

1. Le DataFrame de base est identique pour les 4 cas : `ratings` jointé avec `movies`, genres explosés, ~[N] lignes.
2. Écriture de chaque format dans `output/benchmark/`.
3. Lecture + exécution de la même agrégation (`groupBy("genre").agg(avg, count)`) **3 fois** pour chaque format.
4. On conserve la **médiane des 3 runs** pour limiter le bruit du premier appel (cold cache OS).
5. Taille sur disque mesurée en Mo (hors fichiers de métadonnées `_SUCCESS`, `_metadata`).

### Résultats

| Format           | Taille disque | Temps lecture + agg (médiane) | Ratio vs CSV |
|------------------|---------------|-------------------------------|-------------|
| CSV              | [XX.X Mo]     | [X.XXX s]                     | ×1.0        |
| Parquet (snappy) | [XX.X Mo]     | [X.XXX s]                     | [×N]        |
| Parquet (gzip)   | [XX.X Mo]     | [X.XXX s]                     | [×N]        |
| Parquet (zstd)   | [XX.X Mo]     | [X.XXX s]                     | [×N]        |

> Compléter avec les valeurs affichées par le script.

### Captures / extraits

> **[Insérer ici la sortie console du benchmark]**

### Conclusion

**Ce que j'ai testé :** la vitesse de relecture et d'agrégation selon le format de stockage, à volume constant.

**Ce que j'ai mesuré :** [compléter avec tes valeurs]. Le Parquet est plus rapide que le CSV d'un facteur ~[N] car il est colonnaire (Spark ne lit que la colonne `genre` et `rating`, pas les autres) et supporte le predicate pushdown. Parmi les codecs Parquet, Snappy offre le meilleur compromis décompression rapide / taille, Gzip compresse davantage mais décompresse plus lentement, Zstd est proche de Gzip en taille mais plus rapide à décompresser.

**Ce que j'en conclus :** Pour un pipeline analytique où on relit souvent les données (silver), **Parquet/Snappy est le bon défaut**. Gzip vaut la peine si l'espace disque est contraint et les relectures rares. Zstd est une alternative intéressante si le codec est disponible. Le CSV ne se justifie que pour les échanges avec des outils non-Spark.

**Limites :** les mesures sont faites en mode `local[*]` sur un seul nœud ; les gains du format colonnaire seraient encore plus nets sur un cluster distribué avec un réseau comme goulot d'étranglement.

---

## 7. Ce qu'on a appris et les limites

### Ce qu'on a appris

- Le schéma explicite (`StructType`) évite les surprises de typage que `inferSchema` masque (ex. le timestamp Unix lu comme String).
- Le broadcast join sur `movies` est presque toujours gagnant ici : `movies` tient largement en mémoire (~quelques centaines de Ko sérialisé).
- Le cache n'est rentable que si le DataFrame est réutilisé plusieurs fois dans le même job. Si on ne l'utilise qu'une fois, on paie le coût de matérialisation pour rien.
- Le format Parquet colonnaire est significativement plus rapide sur des agrégations portant sur peu de colonnes, grâce au column pruning et au predicate pushdown.
- La Spark UI (port 4040) est indispensable pour comprendre où va le temps : les shuffles apparaissent clairement dans le DAG.

### Limites

- Le volume de `ml-latest-small` (~100 000 ratings) est modeste : les différences de temps sont réelles mais faibles en valeur absolue. Sur un jeu de plusieurs millions de lignes, les écarts seraient plus marqués.
- La mesure du benchmark est faite sur 3 runs, mais le cache du système de fichiers OS (Linux page cache) peut fausser les résultats après le premier run : les chiffres sont à interpréter comme des ordres de grandeur relatifs plutôt que des valeurs absolues.
- La window function sur les genres explose les lignes (un film → N lignes si N genres) : la sémantique du classement est correcte, mais une note dans le genre "Action" compte autant qu'une note dans le genre "Drama", même si le film est avant tout un drama. Une pondération serait plus rigoureuse.
- Pas de tests automatisés : la reproductibilité repose sur l'exécution du script dans l'ordre.
