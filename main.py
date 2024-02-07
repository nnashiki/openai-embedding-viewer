from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

load_dotenv()
client = OpenAI()

plt.rcParams["font.family"] = "Osaka"
df = pd.read_csv("test_words.csv")


def get_embeddings(words, model="text-embedding-3-small"):
    embeddings = []
    for word in words:
        text = word.replace("\n", " ")
        embedding = client.embeddings.create(input=[text], model=model).data[0].embedding
        embeddings.append(embedding)
    return embeddings


embeddings = get_embeddings(df["word"].tolist())

pca = PCA(n_components=3)
vis_dims = pca.fit_transform(embeddings)
df["embed_vis"] = vis_dims.tolist()

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(projection="3d")
categories = sorted(df["category"].unique())
cmap = plt.get_cmap("tab20")

for i, cat in enumerate(categories):
    sub_matrix = np.array(df[df["category"] == cat]["embed_vis"].tolist())
    x = sub_matrix[:, 0]
    y = sub_matrix[:, 1]
    z = sub_matrix[:, 2]
    colors = [cmap(i / len(categories))] * len(sub_matrix)
    ax.scatter(x, y, z, c=colors, label=cat, depthshade=True)

    # 各データポイントにラベルを付ける
    for word, x_pos, y_pos, z_pos in zip(df[df["category"] == cat]["word"], x, y, z):
        ax.text(x_pos, y_pos, z_pos, word, size=7)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

plt.savefig("word_embeddings_plot_1.png")
plt.show()
