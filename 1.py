import matplotlib
matplotlib.use('Agg')  # 稳定渲染，避免GUI/渲染报错
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# ====================== 全局样式配置（仅保留低版本兼容的参数） ======================
plt.rcParams['font.family'] = 'Times New Roman'  # 论文通用字体
plt.rcParams['font.size'] = 10                   # 期刊标准字号
plt.rcParams['axes.linewidth'] = 0.8             # 坐标轴边框宽度
plt.rcParams['figure.dpi'] = 300                 # 印刷级分辨率
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.unicode_minus'] = False       # 负号正常显示
# 【删除低版本不支持的参数】plt.rcParams['savefig.bbox_inches'] = 'tight'

# ------------------------------ AW-ALouvain算法类 ------------------------------
class AW_ALouvain:
    """自适应权重属性感知Louvain算法（已修复社区映射）"""
    def __init__(self, G, node_attr, max_iter=10, epsilon=1e-4):
        self.original_G = G.copy()
        self.original_nodes = list(G.nodes())
        self.original_attr = node_attr.copy()
        self.G = G.copy()
        self.node_attr = node_attr.copy()
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.n_nodes = len(G.nodes())
        self.n_edges = len(G.edges()) * 2
        
        # 全局属性统计
        self.attr_values = np.array(list(self.original_attr.values()))
        self.mu_A = np.mean(self.attr_values)
        self.sigma_A = np.std(self.attr_values)
        self.omega = self.sigma_A / (self.sigma_A + 1)
        
        # 初始化社区映射
        self.community_map = {node: idx for idx, node in enumerate(self.original_nodes)}
        self.current_community = {node: idx for idx, node in enumerate(G.nodes())}
        self.community_nodes = {idx: [node] for idx, node in enumerate(G.nodes())}
        self.community_attr = {idx: [self.node_attr[node]] for idx, node in enumerate(G.nodes())}

    def _calc_struct_gain(self, node, target_comm):
        k_i = self.G.degree(node)
        k_i_in = sum(1 for neighbor in self.G.neighbors(node) if self.current_community[neighbor] == target_comm)
        sum_tot = sum(self.G.degree(n) for n in self.community_nodes[target_comm])
        delta_q = (k_i_in / self.n_edges) - (sum_tot * k_i) / (self.n_edges **2)
        return delta_q

    def _calc_attr_gain(self, node, target_comm):
        comm_attr = self.community_attr[target_comm]
        mu_C = np.mean(comm_attr) if len(comm_attr) > 0 else self.node_attr[node]
        old_comm = self.current_community[node]
        old_comm_attr = self.community_attr[old_comm]
        mu_old = np.mean(old_comm_attr) if len(old_comm_attr) > 0 else self.node_attr[node]
        
        new_attr_score = np.exp(-((self.node_attr[node] - mu_C)**2) / (2 * self.sigma_A**2)) if self.sigma_A != 0 else 1.0
        old_attr_score = np.exp(-((self.node_attr[node] - mu_old)**2) / (2 * self.sigma_A**2)) if self.sigma_A != 0 else 1.0
        delta_q_attr = new_attr_score - old_attr_score
        delta_q_attr = delta_q_attr / self.n_nodes
        return delta_q_attr

    def _local_optimization(self):
        improved = False
        nodes = list(self.G.nodes())
        for node in nodes:
            old_comm = self.current_community[node]
            neighbor_comms = set(self.current_community[neighbor] for neighbor in self.G.neighbors(node)) - {old_comm}
            if not neighbor_comms:
                continue
            gain_dict = {}
            for comm in neighbor_comms:
                delta_struct = self._calc_struct_gain(node, comm)
                delta_attr = self._calc_attr_gain(node, comm)
                gain = self.omega * delta_struct + (1 - self.omega) * delta_attr
                gain_dict[comm] = gain
            best_comm = max(gain_dict, key=gain_dict.get)
            best_gain = gain_dict[best_comm]
            if best_gain > 0:
                improved = True
                self.current_community[node] = best_comm
                self.community_nodes[old_comm].remove(node)
                self.community_attr[old_comm].remove(self.node_attr[node])
                self.community_nodes[best_comm].append(node)
                self.community_attr[best_comm].append(self.node_attr[node])
        return improved

    def _community_aggregation(self):
        valid_comms = {cid: nodes for cid, nodes in self.community_nodes.items() if len(nodes) > 0}
        super_node_map = {cid: idx for idx, cid in enumerate(valid_comms.keys())}
        new_G = nx.Graph()
        new_G.add_nodes_from(super_node_map.values())
        edge_weights = {}
        for (u, v) in self.G.edges():
            c_u = self.current_community[u]
            c_v = self.current_community[v]
            if c_u == c_v or c_u not in super_node_map or c_v not in super_node_map:
                continue
            sn_u = super_node_map[c_u]
            sn_v = super_node_map[c_v]
            edge_key = tuple(sorted([sn_u, sn_v]))
            edge_weights[edge_key] = edge_weights.get(edge_key, 0) + 1
        for (sn1, sn2), weight in edge_weights.items():
            new_G.add_edge(sn1, sn2, weight=weight)
        new_node_attr = {}
        for cid, sn_id in super_node_map.items():
            new_node_attr[sn_id] = np.mean(self.community_attr[cid])
        # 更新原始节点社区映射
        new_community_map = {}
        for original_node in self.original_nodes:
            for cid, current_nodes in valid_comms.items():
                if original_node in current_nodes:
                    new_community_map[original_node] = super_node_map[cid]
                    break
        self.community_map = new_community_map
        return new_G, new_node_attr

    def _calc_total_modularity(self):
        q_struct = 0.0
        for comm_nodes in self.community_nodes.values():
            if len(comm_nodes) == 0:
                continue
            intra_edges = sum(1 for u in comm_nodes for v in comm_nodes if u < v and self.G.has_edge(u, v)) * 2
            sum_tot = sum(self.G.degree(n) for n in comm_nodes)
            q_struct += (intra_edges / self.n_edges) - (sum_tot / self.n_edges)**2
        q_attr = 0.0
        for comm_attr in self.community_attr.values():
            if len(comm_attr) == 0:
                continue
            mu_C = np.mean(comm_attr)
            attr_score = np.sum(np.exp(-((np.array(comm_attr) - mu_C)**2) / (2 * self.sigma_A**2)) if self.sigma_A != 0 else 1.0)
            q_attr += attr_score / self.n_nodes
        total_q = self.omega * q_struct + (1 - self.omega) * q_attr
        return total_q

    def run(self):
        print(f"=== AW-ALouvain算法启动 ===")
        print(f"自适应权重ω = {self.omega:.4f} | 原始节点数 = {len(self.original_nodes)} | 边数 = {self.n_edges//2}")
        iter_num = 0
        old_q = self._calc_total_modularity()
        while iter_num < self.max_iter:
            print(f"\n迭代次数: {iter_num+1} | 当前模块度: {old_q:.4f}")
            improved = self._local_optimization()
            if not improved:
                print("局部优化无提升，提前终止")
                break
            new_q = self._calc_total_modularity()
            delta_q = new_q - old_q
            print(f"模块度提升: {delta_q:.6f}")
            if delta_q < self.epsilon:
                print(f"模块度提升小于阈值{self.epsilon}，终止迭代")
                break
            self.G, self.node_attr = self._community_aggregation()
            self.current_community = {node: idx for idx, node in enumerate(self.G.nodes())}
            self.community_nodes = {idx: [node] for idx, node in enumerate(self.G.nodes())}
            self.community_attr = {idx: [self.node_attr[node]] for idx, node in enumerate(self.G.nodes())}
            old_q = new_q
            iter_num += 1
        final_comm = self.community_map.copy()
        for node in self.original_nodes:
            if node not in final_comm:
                final_comm[node] = self.current_community.get(node, 0)
        print(f"\n=== 算法结束 ===")
        print(f"最终模块度: {new_q:.4f} | 最终社区数: {len(set(final_comm.values()))}")
        return final_comm

# ------------------------------ 虚假信息传播者识别 ------------------------------
def rumor_spreader_identification(G, community):
    """基于社区结构的虚假信息传播者识别"""
    degree_centrality = {node: G.degree(node) for node in G.nodes()}
    comm_influence = {}
    for cid in set(community.values()):
        comm_nodes = [node for node, c in community.items() if c == cid]
        avg_degree = np.mean([degree_centrality[node] for node in comm_nodes])
        comm_influence[cid] = avg_degree
    risk_score = {}
    for node in G.nodes():
        cid = community[node]
        risk_score[node] = degree_centrality[node] + comm_influence[cid]
    top_k = 5
    sorted_nodes = sorted(risk_score.items(), key=lambda x: x[1], reverse=True)
    high_risk_nodes = [node for node, score in sorted_nodes[:top_k]]
    print("\n=== 虚假信息传播者识别结果 ===")
    print(f"高风险节点（Top-{top_k}）: {high_risk_nodes}")
    for node in high_risk_nodes:
        print(f"节点 {node}: 度中心性={degree_centrality[node]}, 社区影响力={comm_influence[community[node]]:.2f}, 风险评分={risk_score[node]:.2f}")
    return high_risk_nodes, risk_score

# ------------------------------ 论文级可视化 ------------------------------
def plot_paper_visualization(G, final_community, high_risk_nodes, risk_score):
    """生成论文级可视化图（低版本matplotlib兼容）"""
    # 1. 固定布局（保证可复现）
    pos = nx.spring_layout(G, seed=42, k=0.3)
    
    # 2. 社区颜色映射
    comm_ids = [final_community[node] for node in G.nodes()]
    unique_comm = sorted(list(set(comm_ids)))
    n_comm = len(unique_comm)
    color_map = plt.cm.Set3(np.linspace(0, 1, n_comm))
    node_colors = [color_map[unique_comm.index(cid)] for cid in comm_ids]
    
    # 3. 区分普通/高风险节点
    normal_nodes = [n for n in G.nodes() if n not in high_risk_nodes]
    risk_nodes = high_risk_nodes
    
    # 4. 绘制画布
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    # 普通节点
    nx.draw_networkx_nodes(
        G, pos, 
        nodelist=normal_nodes,
        node_color=[node_colors[list(G.nodes()).index(n)] for n in normal_nodes],
        node_size=250,
        edgecolors='black',
        linewidths=0.8,
        ax=ax,
        label='Normal Nodes (Community-colored)'
    )
    
    # 高风险节点（红色粗边框+放大）
    nx.draw_networkx_nodes(
        G, pos, 
        nodelist=risk_nodes,
        node_color=[node_colors[list(G.nodes()).index(n)] for n in risk_nodes],
        node_size=400,
        edgecolors='red',
        linewidths=1.5,
        ax=ax,
        label='High-risk Rumor Spreaders'
    )
    
    # 边（浅灰色弱化）
    nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='gray', ax=ax)
    
    # 仅标注高风险节点
    risk_labels = {n: str(n) for n in risk_nodes}
    normal_labels = {n: "" for n in normal_nodes}
    all_labels = {**risk_labels, **normal_labels}
    nx.draw_networkx_labels(G, pos, labels=all_labels, font_size=9, font_weight='bold', ax=ax)
    
    # 图例与标题
    ax.legend(loc='upper right', frameon=True, edgecolor='black', fontsize=9)
    ax.set_title('Community Detection and Rumor Spreader Identification', fontsize=12, fontweight='bold', pad=15)
    ax.axis('off')
    
    # 图注
    fig.text(0.01, 0.01, 
             'Note: Nodes are colored by community; Red-bordered nodes are top-5 high-risk rumor spreaders.', 
             fontsize=8, ha='left', wrap=True)
    
    # 【关键修复】在savefig中直接指定bbox_inches='tight'（替代全局参数）
    plt.tight_layout()  # 辅助调整布局
    plt.savefig('karate_club_community_rumor.pdf', format='pdf', bbox_inches='tight')  # 论文首选PDF
    plt.savefig('karate_club_community_rumor.png', format='png', bbox_inches='tight', dpi=300)
    print("\n=== 可视化完成 ===")
    print("生成文件：karate_club_community_rumor.pdf (矢量图，推荐插入论文)")
    print("生成文件：karate_club_community_rumor.png (高清位图备用)")

# ------------------------------ 主流程 ------------------------------
if __name__ == "__main__":
    # 1. 加载Karate Club数据集
    G = nx.karate_club_graph()
    node_attr = {node: G.degree(node) for node in G.nodes()}
    
    # 2. 运行AW-ALouvain算法
    aw_louvain = AW_ALouvain(G, node_attr, max_iter=5, epsilon=1e-4)
    final_community = aw_louvain.run()
    
    # 3. 识别高风险传播者
    high_risk_nodes, risk_score = rumor_spreader_identification(G, final_community)
    
    # 4. 输出社区划分结果
    print("\n=== 社区划分结果 ===")
    comm2nodes = {}
    for node, cid in final_community.items():
        if cid not in comm2nodes:
            comm2nodes[cid] = []
        comm2nodes[cid].append(node)
    for cid, nodes in sorted(comm2nodes.items()):
        print(f"社区 {cid}: 节点 {sorted(nodes)}")
    
    # 5. 生成论文级可视化
    plot_paper_visualization(G, final_community, high_risk_nodes, risk_score)