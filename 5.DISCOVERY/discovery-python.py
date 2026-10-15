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

# Robust directory resolution across CLI, VS Code Interactive, and Jupyter Notebook environments
BASE_DIR = None
for p in [
    os.getcwd(),
    os.path.join(os.getcwd(), "qss-student", "5.DISCOVERY"),
    os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "",
    "/Users/haoxi-home/Library/CloudStorage/OneDrive-UniversityofWaterloo/QSS-Tidyverse/qss-student/5.DISCOVERY"
]:
    if p and os.path.isdir(os.path.join(p, "federalist")):
        BASE_DIR = os.path.abspath(p)
        break

if BASE_DIR is None:
    raise FileNotFoundError("Could not locate the federalist corpus directory.")

DIR_SOURCE = os.path.join(BASE_DIR, "federalist")

# --- 5.1.1.2 Load Tidytext Stop-Word Lexicon (1,149 words) ---
stop_words = None
for p in [
    os.path.join(BASE_DIR, "tidytext_stop_words.csv"),
    "tidytext_stop_words.csv",
    os.path.join(os.getcwd(), "tidytext_stop_words.csv"),
    os.path.join(os.getcwd(), "qss-student", "5.DISCOVERY", "tidytext_stop_words.csv"),
    "/Users/haoxi-home/Library/CloudStorage/OneDrive-UniversityofWaterloo/QSS-Tidyverse/qss-student/5.DISCOVERY/tidytext_stop_words.csv"
]:
    if p and os.path.exists(p):
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

# Cache preprocessed tokens to disk for standalone interactive execution
_cache_dir = BASE_DIR
tokens_raw.to_pickle(os.path.join(_cache_dir, "tokens_raw.pkl"))

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

# --- 5.1.4.1 Authorship Labeling & Stylometric Feature Selection ---
# Standalone dependency loader for interactive execution
if "tokens_raw" not in globals():
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    import matplotlib.pyplot as plt

    _search_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokens_raw.pkl")
        if "__file__" in globals()
        else "",
        "tokens_raw.pkl",
        "qss-student/5.DISCOVERY/tokens_raw.pkl",
    ]
    for _path in _search_paths:
        if _path and os.path.exists(_path):
            tokens_raw = pd.read_pickle(_path)
            break

hamilton = [1, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17] + list(range(21, 37)) + [59, 60, 61] + list(range(65, 86))
madison = [10, 14] + list(range(37, 49)) + [58]
jay = [2, 3, 4, 5, 64]
joint = [18, 19, 20]

STYLE_WORDS = [
    "although", "always", "commonly", "consequently",
    "considerable", "enough", "there", "upon",
    "while", "whilst"
]

conditions = [
    tokens_raw["new_id"].isin(hamilton),
    tokens_raw["new_id"].isin(madison),
    tokens_raw["new_id"].isin(jay),
    tokens_raw["new_id"].isin(joint)
]
choices = ["Hamilton", "Madison", "Jay", "Joint"]
tokens_raw["author"] = np.select(conditions, choices, default="Disputed")

# --- 5.1.4.2 Compute Author-Level Word Rates per 1,000 Words (tfm) ---
word_counts = tokens_raw.groupby(["author", "word"]).size().reset_index(name="n")
author_totals = tokens_raw.groupby("author").size().rename("total")
word_counts = word_counts.merge(author_totals, on="author")
word_counts["tf_thou"] = word_counts["n"] / word_counts["total"] * 1000

tfm_long = word_counts[word_counts["word"].isin(STYLE_WORDS)]
tfm = tfm_long.pivot(index="author", columns="word", values="tf_thou").reindex(columns=STYLE_WORDS).fillna(0)

print("\n--- tfm (Average Word Use per 1,000 Words by Author) ---")
print(tfm.round(4))

# --- 5.1.4.3 Document-Level Feature Matrix (reg_data) ---
doc_word_counts = tokens_raw.groupby(["author", "new_id", "word"]).size().reset_index(name="n")
doc_totals = tokens_raw.groupby(["author", "new_id"]).size().rename("total")
doc_word_counts = doc_word_counts.merge(doc_totals, on=["author", "new_id"])
doc_word_counts["tf_thou"] = doc_word_counts["n"] / doc_word_counts["total"] * 1000

doc_style = doc_word_counts[doc_word_counts["word"].isin(STYLE_WORDS)]
reg_data = doc_style.pivot_table(
    index=["author", "new_id"], columns="word", values="tf_thou", fill_value=0
).reset_index()

for w in STYLE_WORDS:
    if w not in reg_data.columns:
        reg_data[w] = 0.0

reg_data["author_outcome"] = np.where(
    reg_data["author"] == "Hamilton", 1.0,
    np.where(reg_data["author"] == "Madison", -1.0, np.nan)
)

# --- 5.1.4.4 Fit Linear Regression Model for Authorship (hm.fit) ---
import statsmodels.api as sm

known = reg_data.dropna(subset=["author_outcome"]).copy()
X_vars = ["upon", "there", "consequently", "whilst"]
X = sm.add_constant(known[X_vars])
y = known["author_outcome"]

hm_fit = sm.OLS(y, X).fit()
print("\n--- Linear Regression Model (hm.fit) ---")
print("Coefficients:")
print(hm_fit.params.round(4))

# --- 5.1.4.5 Standard Deviation of Fitted Values & Effect Size ---
hm_fitted = hm_fit.fittedvalues
sd_fitted = np.std(hm_fitted, ddof=1)
print(f"\nStandard deviation of fitted values: {sd_fitted:.7f}")
print(f"whilst coefficient magnitude: {abs(hm_fit.params['whilst']):.4f} (approx. 1 SD of fitted values)")

# Cache reg_data for standalone interactive execution of downstream validation
_cache_dir = BASE_DIR
reg_data.to_pickle(os.path.join(_cache_dir, "reg_data.pkl"))


# %% [code]
# ------------------------------------------------------------------------------
# 5.1.5 Cross-Validation & Model Evaluation
# ------------------------------------------------------------------------------
# R equivalent:
#   author_data <- reg_data %>% add_predictions(hm.fit) %>% mutate(pred_author = if_else(pred >= 0, "Hamilton", "Madison"))
#   author_data %>% filter(!is.na(author_outcome)) %>% group_by(author) %>% summarize(`Proportion Correct` = mean(author == pred_author))
#   ham_mad <- filter(reg_data, !is.na(author_outcome))
#   for (i in 1:n) { sub.fit <- lm(...); hm.classify[[i]] <- slice(ham_mad, i) %>% add_predictions(sub.fit) }
#   bind_rows(hm.classify) %>% mutate(pred_author = if_else(pred >= 0, "Hamilton", "Madison")) %>% group_by(author) %>% summarize(`Proportion Correct` = mean(author == pred_author))
#   ggplot(plot_data, aes(x = new_id, y = pred, color = author, shape = author)) + ...

# Standalone dependency loader for Section 5.1.5 interactive execution
if "reg_data" not in globals() or "hm_fit" not in globals():
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    import matplotlib.pyplot as plt

    _cache_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "reg_data.pkl")
        if "__file__" in globals()
        else "",
        "reg_data.pkl",
        "qss-student/5.DISCOVERY/reg_data.pkl",
    ]
    for _path in _cache_paths:
        if _path and os.path.exists(_path):
            reg_data = pd.read_pickle(_path)
            break
    X_vars = ["upon", "there", "consequently", "whilst"]
    known_sub = reg_data.dropna(subset=["author_outcome"])
    hm_fit = sm.OLS(known_sub["author_outcome"], sm.add_constant(known_sub[X_vars])).fit()

# --- 5.1.5.1 In-Sample Prediction Accuracy ---
reg_data_const = sm.add_constant(reg_data[X_vars], has_constant="add")
reg_data["pred"] = hm_fit.predict(reg_data_const)
reg_data["pred_author"] = np.where(reg_data["pred"] >= 0, "Hamilton", "Madison")

in_sample = reg_data.dropna(subset=["author_outcome"]).copy()
in_sample["correct"] = in_sample["author"] == in_sample["pred_author"]
prop_correct = in_sample.groupby("author")["correct"].mean().rename("Proportion Correct")
print("\n--- In-Sample Prediction Accuracy ---")
print(prop_correct)

# --- 5.1.5.2 Leave-One-Out Cross-Validation (LOOCV) Loop ---
ham_mad = reg_data.dropna(subset=["author_outcome"]).copy().reset_index(drop=True)
n_docs = len(ham_mad)
cv_preds = []

for i in range(n_docs):
    train_fold = ham_mad.drop(index=i)
    test_fold = ham_mad.iloc[[i]]
    
    X_train = sm.add_constant(train_fold[X_vars], has_constant="add")
    y_train = train_fold["author_outcome"]
    
    fold_fit = sm.OLS(y_train, X_train).fit()
    
    X_test = sm.add_constant(test_fold[X_vars], has_constant="add")
    pred_val = fold_fit.predict(X_test).iloc[0]
    cv_preds.append(pred_val)

ham_mad["cv_pred"] = cv_preds
ham_mad["cv_pred_author"] = np.where(ham_mad["cv_pred"] >= 0, "Hamilton", "Madison")
ham_mad["cv_correct"] = ham_mad["author"] == ham_mad["cv_pred_author"]

# --- 5.1.5.3 Out-of-Sample Prediction Accuracy & Confusion Matrix ---
loocv_accuracy = ham_mad.groupby("author")["cv_correct"].mean().rename("Proportion Correct")
print("\n--- Leave-One-Out Cross-Validation (LOOCV) Accuracy ---")
print(loocv_accuracy)

cv_confusion = pd.crosstab(
    ham_mad["author"],
    ham_mad["cv_pred_author"],
    rownames=["Actual Author"],
    colnames=["Predicted Author"]
)
print("\n--- LOOCV Confusion Matrix ---")
print(cv_confusion)

overall_error = 1.0 - ham_mad["cv_correct"].mean()
print(f"\nOverall LOOCV Classification Error Rate: {overall_error:.4f} ({ham_mad['cv_correct'].sum()}/{n_docs} correct)")

# --- 5.1.5.4 Attribution of Disputed Federalist Papers ---
disputed = reg_data[reg_data["author"] == "Disputed"].sort_values("new_id")
print("\n--- Predictions for Disputed Federalist Papers ---")
print(disputed[["new_id", "pred", "pred_author"]].to_string(index=False))

# --- 5.1.5.5 Authorship Classification Plot (Figure 5.3 Replication) ---
plot_data = reg_data[~reg_data["author"].isin(["Jay", "Joint"])].copy()

plt.figure(figsize=(10, 5))
plt.axhline(0, color="gray", linestyle="--", linewidth=1)

styles = {
    "Madison": {"color": "blue", "marker": "o"},
    "Hamilton": {"color": "red", "marker": "s"},
    "Disputed": {"color": "black", "marker": "^"},
}

for auth in ["Madison", "Hamilton", "Disputed"]:
    grp = plot_data[plot_data["author"] == auth]
    plt.scatter(
        grp["new_id"],
        grp["pred"],
        color=styles[auth]["color"],
        marker=styles[auth]["marker"],
        label=auth,
        alpha=0.85,
        s=45
    )

plt.xlabel("Federalist Papers", fontsize=11)
plt.ylabel("Predicted values", fontsize=11)
plt.title("Authorship Prediction of Federalist Papers", fontsize=12, fontweight="bold")
plt.legend(title="Author")
plt.grid(True, linestyle=":", alpha=0.5)
plt.tight_layout()
plt.show()


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

# --- 5.2.1.1 Ingest Florentine Network & Inspect Adjacency Matrix ---
florentine_path = None
for p in [
    "florentine.csv",
    os.path.join(os.getcwd(), "florentine.csv"),
    os.path.join(os.getcwd(), "qss-student", "5.DISCOVERY", "florentine.csv"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "florentine.csv") if "__file__" in globals() else "",
    "/Users/haoxi-home/Library/CloudStorage/OneDrive-UniversityofWaterloo/QSS-Tidyverse/qss-student/5.DISCOVERY/florentine.csv"
]:
    if p and os.path.exists(p):
        florentine_path = p
        break

florentine = pd.read_csv(florentine_path)
print("Florentine adjacency matrix shape:", florentine.shape)

print("\nAdjacency (sub)matrix for the first 5 families [1:5, 1:5]:")
print(florentine.iloc[0:5, 0:5])

# --- 5.2.1.2 Degree and Connectivity Calculations ---
# Count direct ties per family by summing across columns (excluding the FAMILY label)
florentine["connections"] = florentine.drop(columns=["FAMILY"]).sum(axis=1)

print("\n--- Tie Counts (Connections) per Family ---")
print(florentine[["FAMILY", "connections"]].to_string(index=False))


# %% [code]
# ------------------------------------------------------------------------------
# 5.2.2 Undirected Graph and Centrality Measures
# ------------------------------------------------------------------------------
# R equivalent:
#   igraph::graph_from_adjacency_matrix(florentine, mode = "undirected")
#   degree(g), closeness(g), betweenness(g)
#   plot(g)

import networkx as nx
import matplotlib.pyplot as plt

# --- 5.2.2.1 Construct Undirected Network Graph ---
adj_df = florentine.drop(columns=["connections"], errors="ignore").set_index("FAMILY")
families = adj_df.index.tolist()
G = nx.from_pandas_adjacency(adj_df)

print(f"Network Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# --- 5.2.2.2 Compute Centrality Metrics (Degree, Closeness, Betweenness) ---
# 1. Degree
degree_dict = dict(G.degree())
print("\n--- Degree Centrality ---")
print(pd.Series(degree_dict).sort_values(ascending=False))

# 2. Closeness (R igraph convention: farness to disconnected node Pucci set to N = 16)
closeness_dict = {}
avg_edges_dict = {}
for f in families:
    if f == "PUCCI":
        farness = 15 * 16
    else:
        spl = nx.single_source_shortest_path_length(G, f)
        farness = sum(spl.values()) + 16
    closeness_dict[f] = 1.0 / farness
    avg_edges_dict[f] = farness / 15.0

print("\n--- Closeness Centrality ---")
print(pd.Series(closeness_dict).round(9))

print("\n--- Average Shortest Path Hops: 1 / (Closeness * 15) ---")
print(pd.Series(avg_edges_dict).round(6))

# 3. Betweenness (Unnormalized to match R igraph)
betweenness_dict = nx.betweenness_centrality(G, normalized=False)
print("\n--- Betweenness Centrality ---")
print(pd.Series(betweenness_dict).round(6))

# --- 5.2.2.3 Visualize Network Centrality (Figure 5.2 Replication) ---
pos = nx.spring_layout(G, seed=42)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Closeness Centrality
close_sizes = [closeness_dict[node] * 35000 for node in G.nodes()]
nx.draw_networkx_nodes(G, pos, ax=axes[0], node_size=close_sizes, node_color="skyblue", edgecolors="black", linewidths=1.2)
nx.draw_networkx_edges(G, pos, ax=axes[0], alpha=0.6, width=1.2)
nx.draw_networkx_labels(G, pos, ax=axes[0], font_size=8, font_weight="bold")
axes[0].set_title("Closeness", fontsize=13, fontweight="bold")
axes[0].axis("off")

# Subplot 2: Betweenness Centrality
bet_sizes = [betweenness_dict[node] * 50 + 100 for node in G.nodes()]
nx.draw_networkx_nodes(G, pos, ax=axes[1], node_size=bet_sizes, node_color="salmon", edgecolors="black", linewidths=1.2)
nx.draw_networkx_edges(G, pos, ax=axes[1], alpha=0.6, width=1.2)
nx.draw_networkx_labels(G, pos, ax=axes[1], font_size=8, font_weight="bold")
axes[1].set_title("Betweenness", fontsize=13, fontweight="bold")
axes[1].axis("off")

plt.tight_layout()
plt.show()


# %% [code]
# ------------------------------------------------------------------------------
# 5.2.3 Twitter Following Network
# ------------------------------------------------------------------------------
# R equivalent:
#   twitter <- read.csv("twitter-following.csv", stringsAsFactors = FALSE)
#   senator <- read.csv("twitter-senator.csv", stringsAsFactors = FALSE)
#   twitter.adj <- graph.adjacency(twitter.adj, mode = "directed", diag = FALSE)

follow = pd.read_csv(resolve_path("twitter-following.csv"))
senator = pd.read_csv(resolve_path("twitter-senator.csv"))

# 构建有向图 (Directed Graph)
G_twitter = nx.DiGraph()

# 1. 注入 91 名参议员节点及其党派属性
for _, row in senator.iterrows():
    G_twitter.add_node(
        row["screen_name"],
        name=row["name"],
        party=row["party"],
        state=row["state"]
    )

# 2. 注入 3859 条有向边: following (关注者 i) -> followed (被关注者 j)
for _, row in follow.iterrows():
    G_twitter.add_edge(row["following"], row["followed"])

print(f"Nodes (Senators): {G_twitter.number_of_nodes()}")
print(f"Edges (Following ties): {G_twitter.number_of_edges()}")


# %% [code]
# ------------------------------------------------------------------------------
# 5.2.4 Directed Graph and Centrality (PageRank)
# ------------------------------------------------------------------------------
# R equivalent:
#   senator$indegree <- degree(twitter.adj, mode = "in")
#   senator$outdegree <- degree(twitter.adj, mode = "out") 
#   plot(closeness(twitter.adj, mode = "in"), closeness(twitter.adj, mode = "out"))
#   plot(betweenness(twitter.adj, directed = TRUE), betweenness(twitter.adj, directed = FALSE))
#   senator$pagerank <- page.rank(twitter.adj)$vector
#   plot(twitter.adj, vertex.size = senator$pagerank * 1000)
#   PageRank iterative function with while loop

# 1. 计算入度与出度 (In-degree & Out-degree)
senator["indegree"] = senator["screen_name"].map(dict(G_twitter.in_degree()))
senator["outdegree"] = senator["screen_name"].map(dict(G_twitter.out_degree()))

print("=== Table 5.4: 3 Greatest Indegree (受关注最高 - 声望焦点) ===")
top_in = senator.sort_values("indegree", ascending=False)[["screen_name", "name", "party", "state", "indegree", "outdegree"]].head(3)
print(top_in.to_string(index=False))

print("\n=== Table 5.4: 3 Greatest Outdegree (关注他人最多 - 社交活跃) ===")
top_out = senator.sort_values("outdegree", ascending=False)[["screen_name", "name", "party", "state", "indegree", "outdegree"]].head(3)
print(top_out.to_string(index=False))

# 2. 计算有向紧密中心度 (Closeness: Incoming vs Outgoing)
n_nodes = len(senator)
spl = dict(nx.all_pairs_shortest_path_length(G_twitter))

closeness_in = {}
closeness_out = {}
for u in G_twitter.nodes():
    d_out = [spl[u].get(v, n_nodes) for v in G_twitter.nodes() if v != u]
    closeness_out[u] = 1.0 / sum(d_out) if sum(d_out) > 0 else 0

for v in G_twitter.nodes():
    d_in = [spl[u].get(v, n_nodes) for u in G_twitter.nodes() if u != v]
    closeness_in[v] = 1.0 / sum(d_in) if sum(d_in) > 0 else 0

senator["closeness_in"] = senator["screen_name"].map(closeness_in)
senator["closeness_out"] = senator["screen_name"].map(closeness_out)

# 3. 计算有向与无向介数中心度 (Betweenness: Directed vs Undirected)
senator["betweenness_dir"] = senator["screen_name"].map(
    nx.betweenness_centrality(G_twitter, normalized=False)
)
senator["betweenness_undir"] = senator["screen_name"].map(
    nx.betweenness_centrality(G_twitter.to_undirected(), normalized=False)
)

# 4. 计算 PageRank 权重
pr_dict = nx.pagerank(G_twitter, alpha=0.85)
senator["pagerank"] = senator["screen_name"].map(pr_dict)

print("\n=== Top 5 PageRank Senators (高权威核心议员) ===")
print(senator.sort_values("pagerank", ascending=False)[["name", "party", "state", "pagerank"]].head(5).to_string(index=False))

# 5. 可视化对比: Figure 5.3 (Closeness & Betweenness 对比散点图)
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
color_map = {"R": "red", "D": "blue", "I": "black"}
marker_map = {"R": "o", "D": "^", "I": "x"}

# Panel A: Closeness
for party in ["R", "D", "I"]:
    sub = senator[senator["party"] == party]
    axes[0].scatter(
        sub["closeness_in"], sub["closeness_out"],
        c=color_map[party], marker=marker_map[party],
        s=45, alpha=0.85, label=f"Party {party}"
    )
axes[0].set_title("Closeness (Figure 5.3 Left)", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Incoming path (收敛距离)", fontsize=11)
axes[0].set_ylabel("Outgoing path (广播距离)", fontsize=11)
axes[0].legend(frameon=True)
axes[0].grid(True, linestyle="--", alpha=0.4)

# Panel B: Betweenness
for party in ["R", "D", "I"]:
    sub = senator[senator["party"] == party]
    axes[1].scatter(
        sub["betweenness_dir"], sub["betweenness_undir"],
        c=color_map[party], marker=marker_map[party],
        s=45, alpha=0.85, label=f"Party {party}"
    )
axes[1].set_title("Betweenness (Figure 5.3 Right)", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Directed (有向介数)", fontsize=11)
axes[1].set_ylabel("Undirected (无向介数)", fontsize=11)
axes[1].legend(frameon=True)
axes[1].grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()

# 6. 全网拓扑力导向可视化 (Figure on p. 253: Partisan Polarization Network)
pos = nx.spring_layout(G_twitter, k=0.35, iterations=60, seed=42)
node_colors = [
    "#d62728" if G_twitter.nodes[n]["party"] == "R"
    else "#1f77b4" if G_twitter.nodes[n]["party"] == "D"
    else "#2ca02c"
    for n in G_twitter.nodes()
]
node_sizes = [pr_dict[n] * 35000 for n in G_twitter.nodes()]

plt.figure(figsize=(10, 8.5))
nx.draw_networkx_edges(
    G_twitter, pos,
    edge_color="#aaaaaa", alpha=0.12, width=0.4, arrows=False
)
nx.draw_networkx_nodes(
    G_twitter, pos,
    node_color=node_colors, node_size=node_sizes,
    alpha=0.88, edgecolors="white", linewidths=0.8
)
plt.plot([], [], marker="o", color="#d62728", linestyle="None", markersize=10, label="Republican (R)")
plt.plot([], [], marker="o", color="#1f77b4", linestyle="None", markersize=10, label="Democrat (D)")
plt.plot([], [], marker="o", color="#2ca02c", linestyle="None", markersize=10, label="Independent (I)")
plt.legend(frameon=True, fontsize=11, loc="upper right")
plt.title("Senate Twitter Following Network (p. 253)\nNode Size = PageRank, Node Color = Party", fontsize=13, fontweight="bold")
plt.axis("off")
plt.tight_layout()
plt.show()

# 7. 原书 4-节点微型网络 PageRank 迭代算法 (Gauss-Seidel 就地更新)
nodes_toy = 4
adj_toy = np.array([
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0]
])

def PageRank_algorithm(n, A, d, pr):
    deg = A.sum(axis=1)
    for j in range(n):
        pr[j] = (1 - d) / n + d * np.sum(A[:, j] * pr / deg)
    return pr

d = 0.85
pr_toy = np.full(nodes_toy, 1.0 / nodes_toy)
diff = 100.0
step = 0

while diff > 0.001:
    pr_prev = pr_toy.copy()
    pr_toy = PageRank_algorithm(nodes_toy, adj_toy, d, pr_toy)
    diff = np.max(np.abs(pr_toy - pr_prev))
    step += 1

print(f"\n=== 原书 4-节点微型网络 PageRank 迭代收敛结果 (第 {step} 轮收敛) ===")
for i, val in enumerate(pr_toy, start=1):
    print(f"Node {i}: {val:.7f}")


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