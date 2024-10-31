import math

class BrownClustering:
    def __init__(self, tokens):
        self.tokens = tokens
        self.clusters = {}
        self.cluster_probs = {}
        self.transition_probs = {}
        self.pair_counts = {}
        self.total_words = len(tokens)
        self.cluster_counter = 0

    def initialize_clusters(self):
        unique_tokens = set(self.tokens)
        for word in unique_tokens:
            cluster_id = f"cluster_{self.cluster_counter}"
            self.cluster_counter += 1
            self.clusters[word] = cluster_id
            self.cluster_probs[cluster_id] = self.tokens.count(word) / self.total_words
        print(f"Inicialización: {len(self.clusters)} clusters creados.")

    def calculate_probabilities(self):
        for i in range(len(self.tokens) - 1):
            c_i = self.clusters[self.tokens[i]]
            c_j = self.clusters[self.tokens[i + 1]]
            pair = (c_i, c_j)
            if pair not in self.pair_counts:
                self.pair_counts[pair] = 1
            else:
                self.pair_counts[pair] += 1

        total_bigrams = sum(self.pair_counts.values())
        self.transition_probs = {
            (c_i, c_j): count / total_bigrams for (c_i, c_j), count in self.pair_counts.items()
        }
        print("Probabilidades de transición calculadas.")

    def mutual_information_reduction(self, cluster1, cluster2):
        p_c1 = self.cluster_probs.get(cluster1, 0)
        p_c2 = self.cluster_probs.get(cluster2, 0)
        p_combined = self.transition_probs.get((cluster1, cluster2), 0)
        
        if p_combined > 0 and p_c1 > 0 and p_c2 > 0:
            return p_combined * math.log(p_combined / (p_c1 * p_c2), 2)
        return 0

    def find_best_pair(self):
        best_pair = None
        best_reduction = float('inf')
        
        cluster_list = list(set(self.clusters.values()))
        for i in range(len(cluster_list)):
            for j in range(i + 1, len(cluster_list)):
                cluster1 = cluster_list[i]
                cluster2 = cluster_list[j]
                reduction = self.mutual_information_reduction(cluster1, cluster2)
                if reduction < best_reduction:
                    best_reduction = reduction
                    best_pair = (cluster1, cluster2)
        return best_pair

    def merge_clusters(self, cluster1, cluster2):
        new_cluster = f"cluster_{self.cluster_counter}"
        self.cluster_counter += 1
        new_prob = self.cluster_probs.get(cluster1, 0) + self.cluster_probs.get(cluster2, 0)

        for word in self.clusters:
            if self.clusters[word] == cluster1 or self.clusters[word] == cluster2:
                self.clusters[word] = new_cluster

        self.cluster_probs[new_cluster] = new_prob

        if cluster1 in self.cluster_probs:
            del self.cluster_probs[cluster1]
        if cluster2 in self.cluster_probs:
            del self.cluster_probs[cluster2]

        print(f"Clusters {cluster1} y {cluster2} fusionados en {new_cluster}.")

    def fit(self, target_clusters=100):
        self.initialize_clusters()
        self.calculate_probabilities()
        
        while len(set(self.clusters.values())) > target_clusters:
            cluster1, cluster2 = self.find_best_pair()
            if cluster1 and cluster2:
                self.merge_clusters(cluster1, cluster2)
            else:
                break

        return self.clusters
