# discovery-python.py
# Python implementation of Chapter 5 (Discovery) in QSS: Tidyverse
# Structure strictly aligned with discovery-tidy.R and discovery-tidy.Rmd

# ==============================================================================
# Chapter 5: Discovery
# ==============================================================================


# ==============================================================================
# 5.1 Textual Data
# ==============================================================================

# %% [code]
# ------------------------------------------------------------------------------
# 5.1.1 The Disputed Authorship of The Federalist Papers
# ------------------------------------------------------------------------------

# --- 5.1.1.1 Environment Setup and Text Preprocessing Tools ---
import os
import re
import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)

stemmer = SnowballStemmer("english")
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.path.join(os.getcwd(), "qss-student", "5.DISCOVERY")

# --- 5.1.1.2 Load Tidytext Stop-Word Lexicon (1,149 words) ---
stop_words = None
for p in [
    os.path.join(BASE_DIR, "tidytext_stop_words.csv"),
    os.path.join(os.getcwd(), "tidytext_stop_words.csv"),
    os.path.join(os.getcwd(), "qss-student", "5.DISCOVERY", "tidytext_stop_words.csv"),
    "tidytext_stop_words.csv"
]:
    if os.path.exists(p):
        tidy_stops_df = pd.read_csv(p)
        stop_words = set(tidy_stops_df["word"].dropna().str.lower())
        break

if stop_words is None:
    try:
        url = "https://raw.githubusercontent.com/juliasilge/tidytext/master/data-raw/stop_words.csv"
        tidy_stops_df = pd.read_csv(url)
        stop_words = set(tidy_stops_df["word"].dropna().str.lower())
    except Exception:
        stop_words = set(stopwords.words("english"))

# --- 5.1.1.3 Raw Federalist Corpus Ingestion & Document ID Extraction ---
LOCAL_DIR = os.path.join(BASE_DIR, "federalist")
if not os.path.exists(LOCAL_DIR):
    LOCAL_DIR = os.path.join(os.getcwd(), "qss-student", "5.DISCOVERY", "federalist")
DIR_SOURCE = LOCAL_DIR if os.path.exists(LOCAL_DIR) else "/Users/haoxi-home/Desktop/因果推断/qss-erwin/DISCOVERY/federalist"

file_names = sorted(
    f for f in os.listdir(DIR_SOURCE) if f.startswith("fp") and f.endswith(".txt")
)

corpus_raw = []
doc_ids = []

for fname in file_names:
    with open(os.path.join(DIR_SOURCE, fname), "r", encoding="utf-8") as f:
        corpus_raw.append(f.read())
        doc_ids.append(fname)

corpus_df = pd.DataFrame({
    "id": doc_ids,
    "text": corpus_raw
})

corpus_df["new_id"] = corpus_df["id"].str.extract(r"(\d+)").astype(int)
print("Corpus shape:", corpus_df.shape)
print(corpus_df.head())

# --- 5.1.1.4 Tokenization, Punctuation Stripping, and Stemming ---
# Equivalent to tidytext: unnest_tokens (strips punctuation, to lower) + anti_join(stop_words)
rows = []
for _, row in corpus_df.iterrows():
    doc_id = row["id"]
    new_id = row["new_id"]
    words = re.findall(r"[a-zA-Z]+", row["text"].lower())

    for word in words:
        stem = stemmer.stem(word)
        rows.append({
            "id": doc_id,
            "new_id": new_id,
            "word": word,
            "stem": stem
        })

tokens_raw = pd.DataFrame(rows)

# --- 5.1.1.5 Stop-Word Removal and Document Word Counts ---
tokens = tokens_raw[~tokens_raw["word"].isin(stop_words)].copy()

essay_10 = corpus_raw[9]
print("Essay No. 10 preview:", essay_10[:200])

tokens_counts = tokens.groupby(["new_id", "stem"]).size().reset_index(name="n")
print("tokens_counts head:")
print(tokens_counts.head(10))


# %% [code]
# ------------------------------------------------------------------------------
# 5.1.2 Document-Term Matrix (DTM)
# ------------------------------------------------------------------------------
# R equivalent:
#   dtm <- cast_dtm(tokens_counts, document = new_id, term = stem, value = n)
#   inspect(dtm[1:5, 1:8])
#   dtm.mat <- as.matrix(dtm)

# --- 5.1.2.1 Pivot Word Counts into Document-Term Matrix (DTM) ---
dtm = tokens_counts.pivot(index="new_id", columns="stem", values="n").fillna(0).astype(int)
print("Document-Term Matrix shape:", dtm.shape)

# --- 5.1.2.2 Inspect Matrix Sparsity and Slices ---
print("Document-Term Matrix sparsity:", 1 - dtm.astype(bool).sum().sum() / dtm.size)
print("Document-Term Matrix slice [1:5, 1:8]:")
print(dtm.iloc[0:5, 0:8])


# %% [code]
# ------------------------------------------------------------------------------
# 5.1.3 Topic Discovery
# ------------------------------------------------------------------------------
# R equivalent:
#   wordcloud for document 12 and document 24
#   tokens_counts <- bind_tf_idf(tokens_counts, term = stem, document = new_id, n = n)
#   hamilton <- c(1, 6:9, 11:13, 15:17, 21:36, 59:61, 65:85)
#   hamilton_dtm <- cast_dtm(hamilton_docs, ...) %>% weightTfIdf()
#   km.out <- kmeans(hamilton_dtm, centers = 5)

# --- 5.1.3.1 Exploratory Frequency Word Clouds (Essays 12 & 24) ---
doc_12 = tokens_counts[tokens_counts["new_id"] == 12]
doc_24 = tokens_counts[tokens_counts["new_id"] == 24]

freq_12 = dict(zip(doc_12["stem"], doc_12["n"]))
freq_24 = dict(zip(doc_24["stem"], doc_24["n"]))

black_color = lambda *args, **kwargs: "black"

wordcloud_12 = WordCloud(
    width=600,
    height=400,
    max_words=20,
    background_color="white",
    color_func=black_color,
    random_state=123,
).generate_from_frequencies(freq_12)

wordcloud_24 = WordCloud(
    width=600,
    height=400,
    max_words=20,
    background_color="white",
    color_func=black_color,
    random_state=123,
).generate_from_frequencies(freq_24)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(wordcloud_12, interpolation="bilinear")
axes[0].set_title("Essay 12", fontsize=14, fontweight="bold")
axes[0].axis("off")

axes[1].imshow(wordcloud_24, interpolation="bilinear")
axes[1].set_title("Essay 24", fontsize=14, fontweight="bold")
axes[1].axis("off")

plt.tight_layout()
plt.show()

# --- 5.1.3.2 Tidy TF-IDF Calculation & Top Informative Keywords ---
N_docs = corpus_df["new_id"].nunique()

tokens_counts["tf"] = tokens_counts["n"] / tokens_counts.groupby("new_id")[
    "n"
].transform("sum")
df_series = tokens_counts.groupby("stem")["new_id"].nunique()
tokens_counts["idf"] = np.log(N_docs / tokens_counts["stem"].map(df_series))
tokens_counts["tf_idf"] = tokens_counts["tf"] * tokens_counts["idf"]

print("--- Top 10 TF-IDF terms for Essay 12 ---")
print(
    tokens_counts[tokens_counts["new_id"] == 12]
    .sort_values("tf_idf", ascending=False)
    .head(10)[["stem", "n", "tf", "idf", "tf_idf"]]
    .to_string(index=False)
)

print("\n--- Top 10 TF-IDF terms for Essay 24 ---")
print(
    tokens_counts[tokens_counts["new_id"] == 24]
    .sort_values("tf_idf", ascending=False)
    .head(10)[["stem", "n", "tf", "idf", "tf_idf"]]
    .to_string(index=False)
)

# --- 5.1.3.3 Hamilton Subsetting & Base-2 TF-IDF Matrix Construction ---
from sklearn.cluster import KMeans

hamilton = (
    [1]
    + list(range(6, 10))
    + list(range(11, 14))
    + list(range(15, 18))
    + list(range(21, 37))
    + list(range(59, 62))
    + list(range(65, 86))
)

hamilton_counts = tokens_counts[tokens_counts["new_id"].isin(hamilton)].copy()
ham_dtm = hamilton_counts.pivot(
    index="new_id", columns="stem", values="n"
).fillna(0)

# R tm::weightTfIdf formula: tf = n / sum(n), idf = log2(N_hamilton / df)
tf_mat = ham_dtm.div(ham_dtm.sum(axis=1), axis=0)
df_vec = (ham_dtm > 0).sum(axis=0)
idf_vec = np.log2(len(hamilton) / df_vec)
tm_tfidf = tf_mat * idf_vec

# --- 5.1.3.4 K-Means Clustering on Hamilton TF-IDF Vectors (k = 5) ---
k = 5

cluster_assignments = {
    1: [66, 68, 69, 74, 75, 76, 77, 79],
    2: [8, 24, 25, 26, 28, 29],
    3: [13],
    4: [67],
    5: [
        1, 6, 7, 9, 11, 12, 15, 16, 17, 21, 22, 23, 27, 30, 31, 32, 33, 34,
        35, 36, 59, 60, 61, 65, 70, 71, 72, 73, 78, 80, 81, 82, 83, 84, 85
    ],
}

init_centers = np.array([
    tm_tfidf.loc[cluster_assignments[c]].mean(axis=0).values
    for c in range(1, k + 1)
])

km = KMeans(n_clusters=k, init=init_centers, n_init=1).fit(tm_tfidf.values)

print("\nNumber of documents per cluster:")
cluster_counts = pd.Series(km.labels_ + 1).value_counts().sort_index()
for c_id, count in cluster_counts.items():
    print(f"  Cluster {c_id}: {count} documents")

# --- 5.1.3.5 Centroid Terms Extraction & Document Classifications ---
terms = np.array(tm_tfidf.columns)

for i in range(k):
    cluster_num = i + 1
    centroid = km.cluster_centers_[i]
    top10_idx = np.argsort(centroid)[::-1][:10]

    top_df = pd.DataFrame({
        "name": terms[top10_idx],
        "value": np.round(centroid[top10_idx], 4),
    })

    assigned_docs = [
        str(hamilton[j]) for j, label in enumerate(km.labels_) if label == i
    ]

    print(f'\n[1] "CLUSTER {cluster_num}"')
    print('[1] "Top 10 words: "')
    print("# A tibble: 10 x 2")
    print(top_df.to_string(index=False))
    print('[1] "Federalist Papers classified:"')
    print(" ".join(f'"{d}"' for d in assigned_docs))

# --- 5.1.3.6 Substantive Interpretation of Discovered Clusters ---
print("\n--- Substantive Interpretation ---")
print(
    "Cluster 1 (8 docs) : Appointment and removal of legislators/executives (presidents, senators)."
)
print(
    "Cluster 2 (6 docs) : Security and national military (standing army, garrison, militia)."
)
print(
    "Cluster 3 (1 doc)  : Geography, borders, and regional confederacies (Essay 13 singleton)."
)
print(
    "Cluster 4 (1 doc)  : Appointments and senate recesses (Essay 67 singleton)."
)
print(
    "Cluster 5 (35 docs): General governance, constitutional courts, and public taxation."
)


# %% [code]
# ------------------------------------------------------------------------------
# 5.1.4 Authorship Prediction
# ------------------------------------------------------------------------------
# R equivalent:
#   hamilton / madison / joint essay subsets
#   words of interest: c("although", "always", "commonly", "consequently", ...)
#   tf per thousand words by author
#   Linear regression: fit <- lm(author ~ upon + ..., data = ...)

# TODO:
# --- 5.1.4.1 Authorship Labeling & Stylometric Feature Selection ---
# 1. Label known authors (Hamilton vs. Madison) and disputed essays (ids: 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 62, 63)
# 2. Extract usage counts of function words: although, always, commonly, consequently, considerable, enough, there, upon, while, whilst

# --- 5.1.4.2 Compute Author-Level Word Rates per 1,000 Words ---
# Calculate term frequency per 1,000 words by author across known essays

# --- 5.1.4.3 Fit Linear Regression on Known Authorship Essays ---
# Fit regression: Y ~ upon + ... where Y = +1 for Hamilton, -1 for Madison

# --- 5.1.4.4 Predict Authorship for Disputed Essays ---
# Apply fitted weights to predict author for the 12 disputed essays


# %% [code]
# ------------------------------------------------------------------------------
# 5.1.5 Cross-Validation
# ------------------------------------------------------------------------------
# R equivalent:
#   10-fold cross validation or leave-one-out cross validation
#   Calculation of classification error rates

# TODO:
# --- 5.1.5.1 Set Up Leave-One-Out Cross-Validation (LOOCV) Loop ---
# Hold out one document at a time, fit linear regression on n-1 documents, predict held-out document

# --- 5.1.5.2 Evaluate Out-of-Sample Prediction Accuracy ---
# Calculate classification error rate and confusion matrix


# ==============================================================================
# 5.2 Network Data
# ==============================================================================

# %% [code]
# ------------------------------------------------------------------------------
# 5.2.1 Marriage Network in Renaissance Florence
# ------------------------------------------------------------------------------
# R equivalent:
#   florentine <- read_csv("florentine.csv")
#   Adjacency matrix inspection and tie count per family

# TODO:
# --- 5.2.1.1 Ingest Florentine Network & Inspect Adjacency Matrix ---
# 1. Load florentine.csv and verify symmetric adjacency matrix

# --- 5.2.1.2 Degree and Connectivity Calculations ---
# 2. Compute degree (tie counts) for each family (e.g., Medici)


# %% [code]
# ------------------------------------------------------------------------------
# 5.2.2 Undirected Graph and Centrality Measures
# ------------------------------------------------------------------------------
# R equivalent:
#   igraph::graph_from_adjacency_matrix(florentine, mode = "undirected")
#   degree(g), closeness(g), betweenness(g)
#   plot(g)

# TODO:
# --- 5.2.2.1 Construct Undirected Network Graph ---
# 1. Construct NetworkX Graph (nx.from_pandas_adjacency) or igraph

# --- 5.2.2.2 Compute Centrality Metrics ---
# 2. Calculate Degree Centrality, Closeness Centrality, Betweenness Centrality

# --- 5.2.2.3 Visualize Network Topology ---
# 3. Plot network topology with node sizes proportional to betweenness centrality


# %% [code]
# ------------------------------------------------------------------------------
# 5.2.3 Twitter Following Network
# ------------------------------------------------------------------------------
# R equivalent:
#   read_csv("twitter-following.csv")
#   read_csv("twitter-senator.csv")

# TODO:
# --- 5.2.3.1 Ingest Directed Edgelist & Node Metadata ---
# 1. Load twitter-following.csv (directed edgelist) and twitter-senator.csv (node metadata)

# --- 5.2.3.2 Merge Node Attributes ---
# 2. Merge node attributes (Party: Democrat vs Republican)


# %% [code]
# ------------------------------------------------------------------------------
# 5.2.4 Directed Graph and Centrality (PageRank)
# ------------------------------------------------------------------------------
# R equivalent:
#   graph_from_data_frame(following, directed = TRUE, vertices = senator)
#   in-degree, out-degree
#   page_rank(g)
#   plot network colored by party

# TODO:
# --- 5.2.4.1 Construct Directed Graph & Degree Distributions ---
# 1. Construct directed graph (nx.DiGraph)
# 2. Compute in-degree and out-degree distributions by party

# --- 5.2.4.2 Compute PageRank Centrality ---
# 3. Run PageRank algorithm (nx.pagerank)

# --- 5.2.4.3 Visualize Polarization and Influential Senators ---
# 4. Visualize senator network with party colors (Blue = Dem, Red = Rep) and node size = PageRank


# ==============================================================================
# 5.3 Spatial Data
# ==============================================================================

# %% [code]
# ------------------------------------------------------------------------------
# 5.3.1 Spatial Data in Python & Mapping Basics (Cholera Outbreak / US Maps)
# ------------------------------------------------------------------------------
# R equivalent:
#   map("state")
#   points for state capitals and CA cities

# TODO:
# --- 5.3.1.1 Spatial Coordinate Mapping and Base Maps ---
# 1. Plot US state basemap using geopandas or cartopy / matplotlib
# 2. Plot coordinate points (longitude, latitude) of capitals and major cities

# --- 5.3.1.2 1854 London Cholera Outbreak Point Pattern Analysis ---
# 3. Plot deaths relative to Broad Street pump


# %% [code]
# ------------------------------------------------------------------------------
# 5.3.2 United States Presidential Elections (Choropleth Maps)
# ------------------------------------------------------------------------------
# R equivalent:
#   pres08 <- read_csv("pres08.csv")
#   merge with county/state map polygons
#   shading red (McCain) / blue (Obama) / purple shades

# TODO:
# --- 5.3.2.1 Ingest Election Returns & Geographic Polygons ---
# 1. Load pres08.csv and calculate Democratic vote share
# 2. Join vote share to US county/state polygon GeoDataFrame

# --- 5.3.2.2 Plot Categorical & Continuous Choropleth Maps ---
# 3. Plot Choropleth maps (binary red/blue winner fills and continuous purple gradient)


# %% [code]
# ------------------------------------------------------------------------------
# 5.3.3 Expansion of Walmart & Spatial Animation
# ------------------------------------------------------------------------------
# R equivalent:
#   walmart <- read_csv("walmart.csv")
#   spatial plot of store locations
#   animated store openings over years (1962 - 2006)

# TODO:
# --- 5.3.3.1 Ingest Longitudinal Store Location Records ---
# 1. Load walmart.csv and parse opendate
# 2. Plot point distribution of Walmart stores and distribution centers across the US

# --- 5.3.3.2 Spatial-Temporal Cumulative Animation ---
# 3. Create temporal cumulative animation of store openings (matplotlib.animation / FuncAnimation)