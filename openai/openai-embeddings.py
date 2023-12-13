# %%
from openai.embeddings_utils import get_embeddings

samples = {
    'text': [
        ('사과', 11),
        ('바나나', 11),
        ('복숭아', 11),
        ('배', 11),
        ('편의점', 12),
        ('동그랗다', 13),
        ('껍질', 14),
        ('롯데', 15),
        ('이마트', 12),
        ('마트', 12),
        ('과일', 14),

        ('자동차', 21),
        ('기아', 22),
        ('현대', 22),
        ('BMW', 22),
        ('제네시스', 22),
        ('벤츠', 22),
        ('포드', 22),
        ('로보트', 23),
        ('기계', 23),
        ('알루미늄', 24),
    ],
}
# NOTE: The following code will send a query of batch size 200 to /embeddings
matrix = get_embeddings([x[0] for x in samples['text']], engine="text-embedding-ada-002")

# %%
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
vis_dims = pca.fit_transform(matrix)
samples["embed_vis"] = vis_dims.tolist()

# %%
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(projection='3d')
# ax = fig.add_subplot()
cmap = plt.get_cmap("tab20")

# Plot each sample category individually such that we can set label name.
for i, embed in enumerate(samples['embed_vis']):
    # sub_matrix = np.array(samples[samples["category"] == cat]["embed_vis"].to_list())
    x = embed[0]
    y = embed[1]
    z = embed[2]

    colors = cmap(samples['text'][i][1] - 10)
    ax.scatter(x, y, zs=z, zdir='z', c=colors, label=samples['text'][i][1])
    # ax.scatter(x, y, c=colors, label=samples['text'][i][1])
    # ax.annotate(i + 1, (x, y))

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend(bbox_to_anchor=(1.1, 1))

plt.show()
