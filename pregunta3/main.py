from preprocess import TextProcessor
from BrownClustering import BrownClustering
from Word2Vec import Word2Vec
from Glove import GloVe
if __name__ == "__main__":
    corpus_path = './corpus/eswiki-latest-pages-articles.txt'
    processor = TextProcessor(corpus_path)

    tokens = processor.preprocess_by_batches(batch_size=5000, min_frequency=5, top=1)

    vocab = {word: idx for idx, word in enumerate(set(tokens))}

    # # Word2Vec
    # # Creación y entrenamiento del modelo
    # model = Word2Vec(vocab=vocab, embedding_dim=10, window_size=2, negative_samples=5, learning_rate=0.01, epochs=10)
    # print("Training")
    # model.train(tokens)
    # # Verificar que el embedding sea el mismo
    # embedding = model.get_embedding(tokens[4])
    # print(f"Embedding de la palabra '{tokens[4]}': {embedding}")

    # Glove
    # Crear una instancia de GloVe
    vector_dim = 50
    glove = GloVe(len(set(tokens)), vector_dim)

    # Entrenar el modelo
    window_size = 2
    epochs = 10
    print("Training")
    glove.train(tokens, window_size, epochs)

    # Guardar los embeddings
    glove.save_embeddings("glove_embeddings.txt")


    # # print(tokens_by_batches[:100])
    # brown_clustering = BrownClustering(tokens_by_batches)
    # print(len(tokens_by_batches)) # 5126
    # brown_clustering.initialize_clusters()
    # # print(brown_clustering.clusters)
    # clusters = brown_clustering.fit(target_clusters=50)
    # print(f"Clusters obtenidos: {clusters}")

