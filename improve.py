import matplotlib
# 强制使用Agg后端，解决渲染/保存错误
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

class AW_ALouvain:
    """自适应权重属性感知Louvain算法实现（修复社区映射问题）"""
    def __init__(self, G, node_attr, max_iter=10, epsilon=1e-4):
        """
        初始化参数
        :param G: networkx无向图（社交网络）
        :param node_attr: 字典，{节点id: 属性值}
        :param max_iter: 最大迭代次数
        :param epsilon: 模块度提升阈值（小于该值则终止）
        """
        self.original_G = G.copy()  # 保存原始图（关键：用于最终映射）
        self.original_nodes = list(G.nodes())  # 原始节点列表
        self.original_attr = node_attr.copy()  # 原始节点属性
        
        self.G = G.copy()  # 迭代用图
        self.node_attr = node_attr.copy()
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.n_nodes = len(G.nodes())
        self.n_edges = len(G.edges()) * 2  # 无向图，边数×2（Louvain标准）
        
        # 全局属性统计（基于原始属性，避免聚合后失真）
        self.attr_values = np.array(list(self.original_attr.values()))
        self.mu_A = np.mean(self.attr_values)  # 属性均值
        self.sigma_A = np.std(self.attr_values)  # 属性标准差
        self.omega = self.sigma_A / (self.sigma_A + 1)  # 自适应权重
        
        # 初始化：每个节点独立成社区（原始节点）
        self.community_map = {node: idx for idx, node in enumerate(self.original_nodes)}  # 原始节点→社区id
        self.current_community = {node: idx for idx, node in enumerate(G.nodes())}  # 当前图节点→社区id
        self.community_nodes = {idx: [node] for idx, node in enumerate(G.nodes())}  # 社区id→当前节点列表
        self.community_attr = {idx: [self.node_attr[node]] for idx, node in enumerate(G.nodes())}  # 社区→属性列表

    def _calc_struct_gain(self, node, target_comm):
        """计算结构模块度增益ΔQ_struct（传统Louvain）"""
        k_i = self.G.degree(node)
        k_i_in = sum(1 for neighbor in self.G.neighbors(node) if self.current_community[neighbor] == target_comm)
        sum_tot = sum(self.G.degree(n) for n in self.community_nodes[target_comm])
        delta_q = (k_i_in / self.n_edges) - (sum_tot * k_i) / (self.n_edges **2)
        return delta_q

    def _calc_attr_gain(self, node, target_comm):
        """计算属性一致性模块度增益ΔQ_attr"""
        comm_attr = self.community_attr[target_comm]
        mu_C = np.mean(comm_attr) if len(comm_attr) > 0 else self.node_attr[node]
        old_comm = self.current_community[node]
        old_comm_attr = self.community_attr[old_comm]
        mu_old = np.mean(old_comm_attr) if len(old_comm_attr) > 0 else self.node_attr[node]
        
        new_attr_score = np.exp(-((self.node_attr[node] - mu_C)**2) / (2 * self.sigma_A**2)) if self.sigma_A != 0 else 1.0
        old_attr_score = np.exp(-((self.node_attr[node] - mu_old)**2) / (2 * self.sigma_A**2)) if self.sigma_A != 0 else 1.0
        delta_q_attr = new_attr_score - old_attr_score
        delta_q_attr = delta_q_attr / self.n_nodes  # 归一化
        return delta_q_attr

    def _local_optimization(self):
        """局部优化阶段：迭代移动节点最大化融合增益"""
        improved = False
        nodes = list(self.G.nodes())
        
        for node in nodes:
            old_comm = self.current_community[node]
            neighbor_comms = set(self.current_community[neighbor] for neighbor in self.G.neighbors(node)) - {old_comm}
            if not neighbor_comms:
                continue
            
            # 计算每个邻居社区的融合增益
            gain_dict = {}
            for comm in neighbor_comms:
                delta_struct = self._calc_struct_gain(node, comm)
                delta_attr = self._calc_attr_gain(node, comm)
                gain = self.omega * delta_struct + (1 - self.omega) * delta_attr
                gain_dict[comm] = gain
            
            # 找到最大增益的社区
            best_comm = max(gain_dict, key=gain_dict.get)
            best_gain = gain_dict[best_comm]
            
            # 增益>0则移动节点
            if best_gain > 0:
                improved = True
                # 更新当前社区映射
                self.current_community[node] = best_comm
                # 从原社区移除
                self.community_nodes[old_comm].remove(node)
                self.community_attr[old_comm].remove(self.node_attr[node])
                # 添加到新社区
                self.community_nodes[best_comm].append(node)
                self.community_attr[best_comm].append(self.node_attr[node])
        
        return improved

    def _community_aggregation(self):
        """社区聚合阶段：构建超节点图"""
        # 筛选非空社区
        valid_comms = {cid: nodes for cid, nodes in self.community_nodes.items() if len(nodes) > 0}
        super_node_map = {cid: idx for idx, cid in enumerate(valid_comms.keys())}
        
        # 构建新图
        new_G = nx.Graph()
        new_G.add_nodes_from(super_node_map.values())
        
        # 计算超节点间的边权重
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
        
        # 添加超节点边
        for (sn1, sn2), weight in edge_weights.items():
            new_G.add_edge(sn1, sn2, weight=weight)
        
        # 新节点属性（社区属性均值）
        new_node_attr = {}
        for cid, sn_id in super_node_map.items():
            new_node_attr[sn_id] = np.mean(self.community_attr[cid])
        
        # 更新原始节点的社区映射（关键：聚合后同步更新原始节点的社区ID）
        new_community_map = {}
        for original_node in self.original_nodes:
            # 找到原始节点所属的超节点（当前图节点）
            for cid, current_nodes in valid_comms.items():
                if original_node in current_nodes:
                    new_community_map[original_node] = super_node_map[cid]
                    break
        self.community_map = new_community_map
        
        return new_G, new_node_attr

    def _calc_total_modularity(self):
        """计算当前融合模块度Q = ω*Q_struct + (1-ω)*Q_attr"""
        # 结构模块度Q_struct
        q_struct = 0.0
        for comm_nodes in self.community_nodes.values():
            if len(comm_nodes) == 0:
                continue
            intra_edges = sum(1 for u in comm_nodes for v in comm_nodes if u < v and self.G.has_edge(u, v)) * 2
            sum_tot = sum(self.G.degree(n) for n in comm_nodes)
            q_struct += (intra_edges / self.n_edges) - (sum_tot / self.n_edges)**2
        
        # 属性一致性模块度Q_attr
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
        """运行AW-ALouvain算法主流程"""
        print(f"=== AW-ALouvain算法启动 ===")
        print(f"自适应权重ω = {self.omega:.4f} | 原始节点数 = {len(self.original_nodes)} | 边数 = {self.n_edges//2}")
        
        iter_num = 0
        old_q = self._calc_total_modularity()
        
        while iter_num < self.max_iter:
            print(f"\n迭代次数: {iter_num+1} | 当前模块度: {old_q:.4f}")
            
            # 1. 局部优化
            improved = self._local_optimization()
            if not improved:
                print("局部优化无提升，提前终止")
                break
            
            # 2. 计算新模块度
            new_q = self._calc_total_modularity()
            delta_q = new_q - old_q
            print(f"模块度提升: {delta_q:.6f}")
            
            # 3. 终止判断
            if delta_q < self.epsilon:
                print(f"模块度提升小于阈值{self.epsilon}，终止迭代")
                break
            
            # 4. 社区聚合
            self.G, self.node_attr = self._community_aggregation()
            # 重置当前社区（聚合后的超节点）
            self.current_community = {node: idx for idx, node in enumerate(self.G.nodes())}
            self.community_nodes = {idx: [node] for idx, node in enumerate(self.G.nodes())}
            self.community_attr = {idx: [self.node_attr[node]] for idx, node in enumerate(self.G.nodes())}
            
            old_q = new_q
            iter_num += 1
        
        # 最终社区映射：确保包含所有原始节点
        final_comm = self.community_map.copy()
        # 补充未聚合的节点（防止遗漏）
        for node in self.original_nodes:
            if node not in final_comm:
                final_comm[node] = self.current_community.get(node, 0)
        
        print(f"\n=== 算法结束 ===")
        print(f"最终模块度: {new_q:.4f} | 最终社区数: {len(set(final_comm.values()))}")
        return final_comm

# ------------------------------ 虚假信息传播者识别算法 ------------------------------
def rumor_spreader_identification(G, community):
    """基于社区结构的虚假信息传播者识别"""
    # Step 1: 节点度中心性
    degree_centrality = {node: G.degree(node) for node in G.nodes()}
    
    # Step 2: 社区平均影响力
    comm_influence = {}
    for cid in set(community.values()):
        comm_nodes = [node for node, c in community.items() if c == cid]
        avg_degree = np.mean([degree_centrality[node] for node in comm_nodes])
        comm_influence[cid] = avg_degree
    
    # Step 3: 风险评分
    risk_score = {}
    for node in G.nodes():
        cid = community[node]
        risk_score[node] = degree_centrality[node] + comm_influence[cid]
    
    # Step 4: 排序并返回top-k
    top_k = 5
    sorted_nodes = sorted(risk_score.items(), key=lambda x: x[1], reverse=True)
    high_risk_nodes = [node for node, score in sorted_nodes[:top_k]]
    
    print("\n=== 虚假信息传播者识别结果 ===")
    print(f"高风险节点（Top-{top_k}）: {high_risk_nodes}")
    for node in high_risk_nodes:
        print(f"节点 {node}: 度中心性={degree_centrality[node]}, 社区影响力={comm_influence[community[node]]:.2f}, 风险评分={risk_score[node]:.2f}")
    
    return high_risk_nodes

# ------------------------------ 示例测试 ------------------------------
if __name__ == "__main__":
    # 1. 构造示例社交网络（Karate Club数据集）
    G = nx.karate_club_graph()
    # 为节点添加属性（示例：节点度作为属性）
    node_attr = {node: G.degree(node) for node in G.nodes()}
    
    # 2. 运行AW-ALouvain算法
    aw_louvain = AW_ALouvain(G, node_attr, max_iter=5, epsilon=1e-4)
    final_community = aw_louvain.run()
    
    # 3. 修复颜色生成逻辑（确保与节点数一致）
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10
    plt.figure(figsize=(8, 6), dpi=300)
    
    # 节点颜色：按社区ID分配（确保34个节点都有颜色）
    comm_ids = [final_community[node] for node in G.nodes()]  # 按原始节点顺序取社区ID
    unique_comm = list(set(comm_ids))
    color_map = plt.cm.Set3(np.linspace(0, 1, len(unique_comm)))  # 生成足够的颜色
    colors = [color_map[unique_comm.index(cid)] for cid in comm_ids]  # 每个节点对应颜色
    
    # 绘制网络图
    pos = nx.spring_layout(G, seed=42)  # 固定布局
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=300, edgecolors='black')
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title('AW-ALouvain Community Detection Result (Karate Club)', fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    
    # 保存图片（仅PNG，避免PDF渲染错误；若需PDF先安装ghostscript）
    plt.savefig('aw_louvain_result.png', dpi=300, bbox_inches='tight')
    # 如需PDF，安装ghostscript后取消注释
    # plt.savefig('aw_louvain_result.pdf', format='pdf', bbox_inches='tight')
    
    # 4. 输出社区划分结果
    print("\n=== 社区划分结果 ===")
    comm2nodes = {}
    for node, cid in final_community.items():
        if cid not in comm2nodes:
            comm2nodes[cid] = []
        comm2nodes[cid].append(node)
    
    for cid, nodes in sorted(comm2nodes.items()):
        print(f"社区 {cid}: 节点 {sorted(nodes)}")
    
    # 5. 运行虚假信息传播者识别
    high_risk = rumor_spreader_identification(G, final_community)
    
    # 显示图片（若使用Agg后端，plt.show()会报错，注释即可）
    # plt.show()