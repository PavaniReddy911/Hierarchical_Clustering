# ======================================================
# NEWS TOPIC DISCOVERY STREAMLIT APP
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

import plotly.express as px
import matplotlib.pyplot as plt

# ---------------------------------------------
# PAGE SETUP
# ---------------------------------------------

st.set_page_config(
    page_title="News Topic Discovery",
    layout="wide"
)

st.title("📰 News Topic Discovery (Hierarchical Clustering)")

st.markdown("""
Upload your news dataset (CSV with text)  
This app automatically:
- extracts text
- computes TF-IDF
- shows dendrogram
- clusters articles
- visualizes PCA
- shows cluster summaries
- gives editorial insights
""")

# ---------------------------------------------
# UPLOAD
# ---------------------------------------------

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if not uploaded_file:
    st.info("Please upload a CSV file to begin.")
    st.stop()

df = pd.read_csv(uploaded_file, encoding="latin1", header=None)
df.columns = ["Label", "Text"]

texts = df["Text"].astype(str)

st.write("Dataset preview:", df.head())

# ---------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------

st.sidebar.header("Clustering Settings")

N_CLUSTERS = st.sidebar.slider("Number of Clusters", min_value=2, max_value=12, value=5)
LINKAGE = st.sidebar.selectbox("Linkage Method", ["ward","complete","average","single"])

# ---------------------------------------------
# TF-IDF
# ---------------------------------------------

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=1000,
    ngram_range=(1,2),
    min_df=5
)

X = tfidf.fit_transform(texts)
terms = tfidf.get_feature_names_out()

st.write("TF-IDF matrix shape:", X.shape)

# ---------------------------------------------
# DENDROGRAM (subset)
# ---------------------------------------------

st.write("## 📊 Dendrogram (subset of 400 articles)")

subset = 400 if X.shape[0] > 400 else X.shape[0]
Z = linkage(X[:subset].toarray(), method="ward")

fig, ax = plt.subplots(figsize=(10,4))
dendrogram(Z, ax=ax)
ax.set_title("Dendrogram (use big jumps to choose clusters)")
ax.set_xlabel("Articles")
ax.set_ylabel("Distance")
st.pyplot(fig)

# ---------------------------------------------
# CLUSTERING
# ---------------------------------------------

model = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage=LINKAGE)
labels = model.fit_predict(X.toarray())

st.write(f"## 🎯 Clusters (k={N_CLUSTERS}, linkage={LINKAGE})")

# ---------------------------------------------
# PCA VISUALIZATION
# ---------------------------------------------

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X.toarray())

plot_df = pd.DataFrame({
    "PC1": X_2d[:,0],
    "PC2": X_2d[:,1],
    "Cluster": labels,
    "Snippet": texts.str[:150],
})

fig2 = px.scatter(
    plot_df,
    x="PC1",
    y="PC2",
    color="Cluster",
    hover_data=["Snippet"],
    title="PCA Projection of Clusters"
)
st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------
# CLUSTER SUMMARY
# ---------------------------------------------

st.write("## 📋 Cluster Summary Table")

rows = []
for c in range(N_CLUSTERS):
    idx = np.where(labels == c)[0]
    count = len(idx)

    mean_tfidf = np.asarray(X[idx].mean(axis=0)).ravel()
    top_idx = mean_tfidf.argsort()[-10:][::-1]
    keywords = [terms[i] for i in top_idx]

    snippet = texts.iloc[idx[0]][:200]

    rows.append([c, count, ", ".join(keywords), snippet])

summary_df = pd.DataFrame(
    rows,
    columns=["Cluster ID", "Number of Articles", "Top Keywords", "Representative Snippet"],
)

st.dataframe(summary_df)

# ---------------------------------------------
# VALIDATION
# ---------------------------------------------

st.write("## 🧪 Silhouette Score")

score = silhouette_score(X, labels)

st.write(f"📊 **Silhouette Score:** {score:.4f}")

if score > 0.5:
    st.write("✅ Clusters are well separated")
elif score > 0.25:
    st.write("🟡 Clusters are moderately separated")
elif score > 0:
    st.write("⚠ Clusters overlap moderately")
else:
    st.write("❌ Poor separation (clusters overlap significantly)")

# Linkage comparison
st.write("### Linkage Comparison")

comparisons = []
for m in ["ward","complete","average","single"]:
    model2 = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage=m)
    lab2 = model2.fit_predict(X.toarray())
    s2 = silhouette_score(X, lab2)
    comparisons.append((m, s2))

comp_df = pd.DataFrame(comparisons, columns=["Linkage","Silhouette"])
st.dataframe(comp_df)

# ---------------------------------------------
# BUSINESS INTERPRETATION
# ---------------------------------------------

st.write("## 📰 Editorial Insights")

for cid, count, keyword_str, snippet in rows:
    kw = keyword_str.split(",")[:4]
    st.write(f"🟣 **Cluster {cid}:** Around **{count} articles** focused on themes like **{', '.join(kw)}**")

# ---------------------------------------------
# USER GUIDANCE BOX
# ---------------------------------------------

st.write("""
---
💡 **Insight for Editors:**

Articles grouped in the same cluster share common vocabulary and themes.
These clusters can be used to:

✔ Automatically tag new articles  
✔ Power recommendation systems  
✔ Improve content organization  
✔ Help editors find related stories faster  
""")
# ======================================================
# NEWS TOPIC DISCOVERY STREAMLIT APP
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

import plotly.express as px
import matplotlib.pyplot as plt

# ---------------------------------------------
# PAGE SETUP
# ---------------------------------------------

st.set_page_config(
    page_title="News Topic Discovery",
    layout="wide"
)

st.title("📰 News Topic Discovery (Hierarchical Clustering)")

st.markdown("""
Upload your news dataset (CSV with text)  
This app automatically:
- extracts text
- computes TF-IDF
- shows dendrogram
- clusters articles
- visualizes PCA
- shows cluster summaries
- gives editorial insights
""")

# ---------------------------------------------
# UPLOAD
# ---------------------------------------------

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if not uploaded_file:
    st.info("Please upload a CSV file to begin.")
    st.stop()

df = pd.read_csv(uploaded_file, encoding="latin1", header=None)
df.columns = ["Label", "Text"]

texts = df["Text"].astype(str)

st.write("Dataset preview:", df.head())

# ---------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------

st.sidebar.header("Clustering Settings")

N_CLUSTERS = st.sidebar.slider("Number of Clusters", min_value=2, max_value=12, value=5)
LINKAGE = st.sidebar.selectbox("Linkage Method", ["ward","complete","average","single"])

# ---------------------------------------------
# TF-IDF
# ---------------------------------------------

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=1000,
    ngram_range=(1,2),
    min_df=5
)

X = tfidf.fit_transform(texts)
terms = tfidf.get_feature_names_out()

st.write("TF-IDF matrix shape:", X.shape)

# ---------------------------------------------
# DENDROGRAM (subset)
# ---------------------------------------------

st.write("## 📊 Dendrogram (subset of 400 articles)")

subset = 400 if X.shape[0] > 400 else X.shape[0]
Z = linkage(X[:subset].toarray(), method="ward")

fig, ax = plt.subplots(figsize=(10,4))
dendrogram(Z, ax=ax)
ax.set_title("Dendrogram (use big jumps to choose clusters)")
ax.set_xlabel("Articles")
ax.set_ylabel("Distance")
st.pyplot(fig)

# ---------------------------------------------
# CLUSTERING
# ---------------------------------------------

model = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage=LINKAGE)
labels = model.fit_predict(X.toarray())

st.write(f"## 🎯 Clusters (k={N_CLUSTERS}, linkage={LINKAGE})")

# ---------------------------------------------
# PCA VISUALIZATION
# ---------------------------------------------

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X.toarray())

plot_df = pd.DataFrame({
    "PC1": X_2d[:,0],
    "PC2": X_2d[:,1],
    "Cluster": labels,
    "Snippet": texts.str[:150],
})

fig2 = px.scatter(
    plot_df,
    x="PC1",
    y="PC2",
    color="Cluster",
    hover_data=["Snippet"],
    title="PCA Projection of Clusters"
)
st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------
# CLUSTER SUMMARY
# ---------------------------------------------

st.write("## 📋 Cluster Summary Table")

rows = []
for c in range(N_CLUSTERS):
    idx = np.where(labels == c)[0]
    count = len(idx)

    mean_tfidf = np.asarray(X[idx].mean(axis=0)).ravel()
    top_idx = mean_tfidf.argsort()[-10:][::-1]
    keywords = [terms[i] for i in top_idx]

    snippet = texts.iloc[idx[0]][:200]

    rows.append([c, count, ", ".join(keywords), snippet])

summary_df = pd.DataFrame(
    rows,
    columns=["Cluster ID", "Number of Articles", "Top Keywords", "Representative Snippet"],
)

st.dataframe(summary_df)

# ---------------------------------------------
# VALIDATION
# ---------------------------------------------

st.write("## 🧪 Silhouette Score")

score = silhouette_score(X, labels)

st.write(f"📊 **Silhouette Score:** {score:.4f}")

if score > 0.5:
    st.write("✅ Clusters are well separated")
elif score > 0.25:
    st.write("🟡 Clusters are moderately separated")
elif score > 0:
    st.write("⚠ Clusters overlap moderately")
else:
    st.write("❌ Poor separation (clusters overlap significantly)")

# Linkage comparison
st.write("### Linkage Comparison")

comparisons = []
for m in ["ward","complete","average","single"]:
    model2 = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage=m)
    lab2 = model2.fit_predict(X.toarray())
    s2 = silhouette_score(X, lab2)
    comparisons.append((m, s2))

comp_df = pd.DataFrame(comparisons, columns=["Linkage","Silhouette"])
st.dataframe(comp_df)

# ---------------------------------------------
# BUSINESS INTERPRETATION
# ---------------------------------------------

st.write("## 📰 Editorial Insights")

for cid, count, keyword_str, snippet in rows:
    kw = keyword_str.split(",")[:4]
    st.write(f"🟣 **Cluster {cid}:** Around **{count} articles** focused on themes like **{', '.join(kw)}**")

# ---------------------------------------------
# USER GUIDANCE BOX
# ---------------------------------------------

st.write("""
---
💡 **Insight for Editors:**

Articles grouped in the same cluster share common vocabulary and themes.
These clusters can be used to:

✔ Automatically tag new articles  
✔ Power recommendation systems  
✔ Improve content organization  
✔ Help editors find related stories faster  
""")
