from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, IntegerType, FloatType, LongType, StringType
)
from pyspark.sql.window import Window
import time
import os

spark = (
    SparkSession.builder
    .appName("MovieLens-Pipeline")
    .master("local[*]")
    # Active l'AQE (utile pour l'exploration)
    .config("spark.sql.adaptive.enabled", "true")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

DATA_DIR    = "data/ml-latest-small"         # dossier des CSV MovieLens
SILVER_DIR  = "output/silver"                # couche nettoyée en Parquet
GOLD_DIR    = "output/gold"                  # résultats des analyses
BENCH_DIR   = "output/benchmark"             # exploration formats

os.makedirs(SILVER_DIR, exist_ok=True)
os.makedirs(GOLD_DIR,   exist_ok=True)
os.makedirs(BENCH_DIR,  exist_ok=True)

# ÉTAPE 1 — INGESTION & NETTOYAGE  (Bronze → Silver)

print("\n=== Ingestion et Nettoyage ===")

schema_ratings = StructType([
    StructField("userId",    IntegerType(), True),
    StructField("movieId",   IntegerType(), True),
    StructField("rating",    FloatType(),   True),
    StructField("timestamp", LongType(),    True),
])

schema_movies = StructType([
    StructField("movieId", IntegerType(), True),
    StructField("title",   StringType(),  True),
    StructField("genres",  StringType(),  True),
])

schema_tags = StructType([
    StructField("userId",    IntegerType(), True),
    StructField("movieId",   IntegerType(), True),
    StructField("tag",       StringType(),  True),
    StructField("timestamp", LongType(),    True),
])

ratings_raw = spark.read.csv(
    f"{DATA_DIR}/ratings.csv", schema=schema_ratings, header=True
)
movies_raw = spark.read.csv(
    f"{DATA_DIR}/movies.csv", schema=schema_movies, header=True
)
tags_raw = spark.read.csv(
    f"{DATA_DIR}/tags.csv", schema=schema_tags, header=True
)

print("Schéma ratings :")
ratings_raw.printSchema()

print(f"ratings brut  : {ratings_raw.count()} lignes")
print(f"movies brut   : {movies_raw.count()} lignes")
print(f"tags brut     : {tags_raw.count()} lignes")

# Test rapide de la structure
ratings_raw.describe("rating").show()
ratings_raw.show(5)

# Nettoyage ratings
ratings = (
    ratings_raw
    # Supprimer les doublons
    .dropDuplicates()
    # Supprimer les lignes avec valeurs manquantes critiques
    .dropna(subset=["userId", "movieId", "rating"])
    # Garder seulement les notes valides (0.5 à 5.0 par paliers de 0.5)
    .filter((F.col("rating") >= 0.5) & (F.col("rating") <= 5.0))
    # Convertir timestamp Unix en date lisible
    .withColumn("rating_date", F.to_date(F.from_unixtime(F.col("timestamp"))))
    .withColumn("rating_year", F.year(F.col("rating_date")))
    .drop("timestamp")
)

# Nettoyage movies
movies = (
    movies_raw
    .dropDuplicates(["movieId"])
    .dropna(subset=["movieId", "title"])
    # Extraire l'année du titre  ex: "Toy Story (1995)" -> 1995
    .withColumn(
        "release_year",
        F.when(
        F.regexp_extract(F.col("title"), r"\((\d{4})\)", 1) != "",
        F.regexp_extract(F.col("title"), r"\((\d{4})\)", 1).cast(IntegerType())
    ).otherwise(None)
    )
    # Nettoyer le titre (sans l'année)
    .withColumn(
        "title_clean",
        F.regexp_replace(F.col("title"), r"\s*\(\d{4}\)\s*$", "")
    )
    # Garder les films avec une année cohérente (1888 premier film de l'histoire)
    .filter(
        F.col("release_year").isNull() |
        ((F.col("release_year") >= 1888) & (F.col("release_year") <= 2024))
    )
)

# Bilan du nettoyage
n_ratings_brut   = ratings_raw.count()
n_ratings_silver = ratings.count()
n_movies_brut    = movies_raw.count()
n_movies_silver  = movies.count()

print(f"\nRatings : {n_ratings_brut} → {n_ratings_silver} "
      f"({n_ratings_brut - n_ratings_silver} lignes écartées)")
print(f"Movies  : {n_movies_brut} → {n_movies_silver} "
      f"({n_movies_brut - n_movies_silver} lignes écartées)")

# Écriture Silver en Parquet, partitionné par année de note
(
    ratings
    .write
    .mode("overwrite")
    .partitionBy("rating_year")
    .parquet(f"{SILVER_DIR}/ratings")
)
(
    movies
    .write
    .mode("overwrite")
    .parquet(f"{SILVER_DIR}/movies")
)
print(f"\nCouche Silver écrite dans {SILVER_DIR}/")

# ÉTAPE 2 — ANALYSES  (Silver → Gold)

print("\n=== ÉTAPE 2 : Analyses ===")

# Relire depuis la couche Silver (pas le brut)
ratings_s = spark.read.parquet(f"{SILVER_DIR}/ratings")
movies_s  = spark.read.parquet(f"{SILVER_DIR}/movies")

# ── ANALYSE 1 : Films les mieux notés (agrégation avec seuil de votes) ──
print("\n--- Analyse 1 : Films les mieux notés ---")

# Broadcast de movies (petit) pour éviter le shuffle lors de la jointure
movies_bc = F.broadcast(movies_s)

top_movies = (
    ratings_s
    .groupBy("movieId")
    .agg(
        F.count("rating").alias("nb_votes"),
        F.round(F.avg("rating"), 2).alias("avg_rating"),
    )
    # Seuil de crédibilité : au moins 50 votes
    .filter(F.col("nb_votes") >= 50)
    # Jointure avec les titres (broadcast join)
    .join(movies_bc, on="movieId", how="left")
    .select("title_clean", "genres", "nb_votes", "avg_rating")
    .orderBy(F.desc("avg_rating"), F.desc("nb_votes"))
)

print("Top 10 films les mieux notés (≥ 50 votes) :")
top_movies.show(10, truncate=False)

top_movies.write.mode("overwrite").parquet(f"{GOLD_DIR}/top_movies")
print(f"Résultat écrit dans {GOLD_DIR}/top_movies")

# ── ANALYSE 2 : Popularité par genre (jointure + agrégation) ──
print("\n--- Analyse 2 : Popularité par genre ---")

# Exploser les genres (un film peut avoir plusieurs genres)
ratings_with_genres = (
    ratings_s
    .join(F.broadcast(movies_s.select("movieId", "genres")), on="movieId", how="inner")
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

print("Popularité par genre :")
genre_stats.show(20, truncate=False)

genre_stats.write.mode("overwrite").parquet(f"{GOLD_DIR}/genre_stats")
print(f"Résultat écrit dans {GOLD_DIR}/genre_stats")

# ── ANALYSE 3 : Classement des films par genre (window function) ──
print("\n--- Analyse 3 : Classement des films par genre (window function) ---")

# Films avec suffisamment de votes pour être classés
rated_movies = (
    ratings_s
    .groupBy("movieId")
    .agg(
        F.count("rating").alias("nb_votes"),
        F.round(F.avg("rating"), 2).alias("avg_rating"),
    )
    .filter(F.col("nb_votes") >= 20)
    .join(F.broadcast(movies_s.select("movieId", "title_clean", "genres")), on="movieId")
    .withColumn("genre", F.explode(F.split(F.col("genres"), "\\|")))
    .filter(F.col("genre") != "(no genres listed)")
)

# Window : rang par genre selon la note moyenne
window_genre = Window.partitionBy("genre").orderBy(F.desc("avg_rating"))

top_by_genre = (
    rated_movies
    .withColumn("rank_in_genre", F.rank().over(window_genre))
    .filter(F.col("rank_in_genre") <= 3)   # Top 3 par genre
    .select("genre", "rank_in_genre", "title_clean", "avg_rating", "nb_votes")
    .orderBy("genre", "rank_in_genre")
)

print("Top 3 films par genre :")
top_by_genre.show(30, truncate=False)

top_by_genre.write.mode("overwrite").parquet(f"{GOLD_DIR}/top_by_genre")
print(f"Résultat écrit dans {GOLD_DIR}/top_by_genre")

# ÉTAPE 3 — OPTIMISATION MESURÉE

# Sans broadcast (join standard — Spark décide seul)
print("\nJointure SANS hint broadcast :")
t0 = time.time()
(
    ratings_s
    .join(movies_s.select("movieId", "genres"), on="movieId")
    .groupBy("genres")
    .agg(F.count("*").alias("n"))
    .write.mode("overwrite").parquet(f"{GOLD_DIR}/tmp_no_broadcast")
)
t_no_bc = time.time() - t0
print(f"  → Temps sans broadcast : {t_no_bc:.2f}s")

# Avec broadcast explicite sur movies (petit fichier ~9 000 lignes)
print("\nJointure AVEC hint broadcast :")

t0 = time.time()
(
    ratings_s
    .join(F.broadcast(movies_s.select("movieId", "genres")), on="movieId")
    .groupBy("genres")
    .agg(F.count("*").alias("n"))
    .write.mode("overwrite").parquet(f"{GOLD_DIR}/tmp_broadcast")
)
t_bc = time.time() - t0
print(f"  → Temps avec broadcast : {t_bc:.2f}s")
print(f"  → Gain : {t_no_bc - t_bc:.2f}s  ({(t_no_bc - t_bc) / t_no_bc * 100:.1f}%)")

# ── 3b. Cache d'un DataFrame réutilisé ──
print("\nCache d'un DataFrame réutilisé :")

# ratings_with_genres est utilisé dans plusieurs analyses → mise en cache
ratings_cached = (
    ratings_s
    .join(F.broadcast(movies_s.select("movieId", "genres", "title_clean")), on="movieId")
    .withColumn("genre", F.explode(F.split(F.col("genres"), "\\|")))
    .filter(F.col("genre") != "(no genres listed)")
    .cache()
)

t0 = time.time()
ratings_cached.count()   # Force la matérialisation du cache
t_cache_fill = time.time() - t0
print(f"  → Temps pour remplir le cache : {t_cache_fill:.2f}s")

# Requête 1 sur le DataFrame caché
t0 = time.time()
ratings_cached.groupBy("genre").agg(F.avg("rating")).collect()
t_cached_1 = time.time() - t0
print(f"  → Requête 1 sur cache : {t_cached_1:.3f}s")

# Requête 2 sur le même DataFrame caché
t0 = time.time()
ratings_cached.groupBy("genre").agg(F.count("*")).collect()
t_cached_2 = time.time() - t0
print(f"  → Requête 2 sur cache : {t_cached_2:.3f}s  (relecture depuis mémoire)")

ratings_cached.unpersist()

# ÉTAPE 4 — EXPLORATION : BENCHMARK DE FORMATS

print("\n=== ÉTAPE 4 : Exploration — Benchmark de formats ===")

# On compare 4 combinaisons : CSV / Parquet (snappy) / Parquet (gzip) / Parquet (zstd)
# Sur la même agrégation : note moyenne par genre

ratings_full = spark.read.parquet(f"{SILVER_DIR}/ratings")
movies_full  = spark.read.parquet(f"{SILVER_DIR}/movies")

base_df = (
    ratings_full
    .join(F.broadcast(movies_full.select("movieId", "genres")), on="movieId")
    .withColumn("genre", F.explode(F.split(F.col("genres"), "\\|")))
    .filter(F.col("genre") != "(no genres listed)")
)

def agg_query(df):
    """Agrégation de référence : note moyenne par genre."""
    return df.groupBy("genre").agg(
        F.round(F.avg("rating"), 3).alias("avg_rating"),
        F.count("*").alias("n")
    ).orderBy("genre")

# ── Écriture des 4 formats ──
formats = [
    ("csv",           {"format": "csv",     "options": {"header": "true"}}),
    ("parquet_snappy",{"format": "parquet",  "options": {"compression": "snappy"}}),
    ("parquet_gzip",  {"format": "parquet",  "options": {"compression": "gzip"}}),
    ("parquet_zstd",  {"format": "parquet",  "options": {"compression": "zstd"}}),
]

print("\nÉcriture des 4 formats :")
for name, cfg in formats:
    path = f"{BENCH_DIR}/{name}"
    t0 = time.time()
    writer = base_df.write.mode("overwrite").format(cfg["format"])
    for k, v in cfg["options"].items():
        writer = writer.option(k, v)
    writer.save(path)
    t_write = time.time() - t0
    # Taille sur disque (en Mo)
    size_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, dn, fn in os.walk(path)
        for f in fn
        if not f.startswith("_")
    ) / (1024 * 1024)
    print(f"  {name:<20} : écriture {t_write:.2f}s  |  taille {size_mb:.2f} Mo")

    # ── Lecture + agrégation depuis chaque format ──
print("\nLecture + agrégation depuis chaque format :")
results_bench = []
for name, cfg in formats:
    path = f"{BENCH_DIR}/{name}"
    # Mesure sur 3 runs, on garde la médiane
    times = []
    for _ in range(3):
        t0 = time.time()
        reader = spark.read.format(cfg["format"])
        for k, v in cfg["options"].items():
            reader = reader.option(k, v)
        df = reader.load(path)
        agg_query(df).write.mode("overwrite").format("noop").save()
        times.append(time.time() - t0)
    median_t = sorted(times)[1]
    results_bench.append((name, median_t))
    print(f"  {name:<20} : {median_t:.3f}s  (médiane de 3 runs)")

print("\nRésumé benchmark :")
baseline = results_bench[0][1]
for name, t in results_bench:
    ratio = baseline / t
    print(f"  {name:<20} : {t:.3f}s  (x{ratio:.1f} vs CSV)")

input(">>> Appuie sur Entrée pour fermer la Spark UI...")
spark.stop()

