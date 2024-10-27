from preprocess import TextProcessor
from BrownClustering import BrownClustering
if __name__ == "__main__":
    corpus_path = './corpus/eswiki-latest-pages-articles.txt'
    processor = TextProcessor(corpus_path)

    tokens_by_batches = processor.preprocess_by_batches(batch_size=5000, min_frequency=5, top=1)

    # print(tokens_by_batches[:100])
    brown_clustering = BrownClustering(tokens_by_batches)
    print(len(tokens_by_batches)) # 5126
    brown_clustering.initialize_clusters()
    # print(brown_clustering.clusters)
    clusters = brown_clustering.fit(target_clusters=50)
    print(f"Clusters obtenidos: {clusters}")

