import networkx as nx
import numpy as np
import time
import matplotlib.pyplot as plt
import community
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score

# =========================
# 1. 数据集与真实标签
# =========================
G = nx.karate_club_graph()

true_labels = np.array([
    0 if G.nodes[i]['club'] == 'Mr. Hi' else 1
    for i in G.nodes()
])

# =========================
# 2. 工具函数
# =========================
def labels_from_communities(communities, n):
    labels = np.zeros(n, dtype=int)
    for cid, comm in enumerate(communities):
        for node in comm:
            labels[node] = cid
    return labels

def best_gn_partition(G, max_groups=6):
    best_mod = -1
    best_comm = None
    comp = nx.community.girvan_newman(G)
    for communities in comp:
        comm = [list(c) for c in communities]
        mod = nx.community.modularity(G, comm)
        if mod > best_mod:
            best_mod = mod
            best_comm = comm
        if len(comm) >= max_groups:
            break
    return best_comm, best_mod

# =========================
# 3. 算法评测函数
# =========================
def evaluate_algorithms(G, runs=5):
    results = []

    # -------- Louvain --------
    mods, nmis, times = [], [], []
    for _ in range(runs):
        start = time.time()
        part = community.best_partition(G, random_state=42)
        comm = {}
        for node, cid in part.items():
            comm.setdefault(cid, []).append(node)
        comm = list(comm.values())
        t = time.time() - start

        mods.append(nx.community.modularity(G, comm))
        nmis.append(normalized_mutual_info_score(true_labels,
                                                 labels_from_communities(comm, G.number_of_nodes())))
        times.append(t)
    results.append(("Louvain", np.mean(mods), np.mean(nmis), np.mean(times)))

    # -------- Girvan–Newman --------
    start = time.time()
    gn_comm, gn_mod = best_gn_partition(G)
    t = time.time() - start
    gn_labels = labels_from_communities(gn_comm, G.number_of_nodes())
    gn_nmi = normalized_mutual_info_score(true_labels, gn_labels)
    results.append(("GN", gn_mod, gn_nmi, t))

    # -------- Spectral Clustering --------
    mods, nmis, times = [], [], []
    adj = nx.to_numpy_array(G)
    for _ in range(runs):
        start = time.time()
        sc = SpectralClustering(
            n_clusters=2,
            affinity='precomputed',
            eigen_solver='arpack',
            n_init=1,
            random_state=42
        )
        labels = sc.fit_predict(adj)
        t = time.time() - start

        comm = [[] for _ in range(2)]
        for node, cid in enumerate(labels):
            comm[cid].append(node)

        mods.append(nx.community.modularity(G, comm))
        nmis.append(normalized_mutual_info_score(true_labels, labels))
        times.append(t)
    results.append(("Spectral", np.mean(mods), np.mean(nmis), np.mean(times)))

    # -------- Label Propagation --------
    start = time.time()
    lp_comm = [list(c) for c in nx.community.label_propagation_communities(G)]
    t = time.time() - start
    lp_labels = labels_from_communities(lp_comm, G.number_of_nodes())
    lp_mod = nx.community.modularity(G, lp_comm)
    lp_nmi = normalized_mutual_info_score(true_labels, lp_labels)
    results.append(("Label Propagation", lp_mod, lp_nmi, t))

    # -------- K-clique (k=3) --------
    start = time.time()
    raw = list(nx.community.k_clique_communities(G, k=3))
    node2cid = {}
    for cid, comm in enumerate(raw):
        for node in comm:
            if node not in node2cid:
                node2cid[node] = cid
    for node in G.nodes():
        node2cid.setdefault(node, len(raw))

    comm = {}
    for node, cid in node2cid.items():
        comm.setdefault(cid, []).append(node)
    comm = list(comm.values())
    t = time.time() - start
    labels = np.array([node2cid[i] for i in range(G.number_of_nodes())])
    results.append((
        "K-clique",
        nx.community.modularity(G, comm),
        normalized_mutual_info_score(true_labels, labels),
        t
    ))

    return results

# =========================
# 4. 可视化表格函数
# =========================
def plot_results_table(results, filename="algorithm_results.png"):
    columns = ["Algorithm", "Modularity", "NMI", "Runtime (s)"]
    cell_text = []
    for r in results:
        cell_text.append([
            r[0],
            f"{r[1]:.4f}",
            f"{r[2]:.4f}",
            f"{r[3]:.4f}"
        ])
    fig, ax = plt.subplots(figsize=(8, 2 + len(results)*0.4))
    ax.axis('off')
    table = ax.table(cellText=cell_text, colLabels=columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.show()

# =========================
# 5. 运行实验并绘制表格
# =========================
results = evaluate_algorithms(G, runs=5)

# 打印结果
print("=" * 90)
print(f"{'Algorithm':<20} {'Modularity':<15} {'NMI':<15} {'Runtime (s)':<15}")
print("=" * 90)
for r in results:
    print(f"{r[0]:<20} {r[1]:<15.4f} {r[2]:<15.4f} {r[3]:<15.4f}")

# 绘制可视化表格
plot_results_table(results)
