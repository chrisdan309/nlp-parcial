import math
import random
from collections import defaultdict

class GloVe:
    def __init__(self, vocab_size, vector_dim, alpha=0.75, x_max=100, learning_rate=0.05):
        self.vocab_size = vocab_size
        self.vector_dim = vector_dim
        self.alpha = alpha
        self.x_max = x_max
        self.learning_rate = learning_rate
        
        self.co_occurrence_matrix = [[0] * vocab_size for _ in range(vocab_size)]
        self.word_vectors = [[random.random() for _ in range(vector_dim)] for _ in range(vocab_size)]
        self.biases = [random.random() for _ in range(vocab_size)]
        self.word_to_index = {}
        self.index_to_word = {}

    def build_vocab(self, tokens):
        unique_words = set(tokens)
        self.vocab_size = len(unique_words)
        self.word_to_index = {word: i for i, word in enumerate(unique_words)}
        self.index_to_word = {i: word for i, word in enumerate(unique_words)}

    def build_co_occurrence_matrix(self, tokens, window_size):
        for i, word in enumerate(tokens):
            if word in self.word_to_index:
                current_word_index = self.word_to_index[word]
                for j in range(max(0, i - window_size), min(len(tokens), i + window_size + 1)):
                    if j != i and tokens[j] in self.word_to_index:
                        context_word_index = self.word_to_index[tokens[j]]
                        self.co_occurrence_matrix[current_word_index][context_word_index] += 1

    def cost_function(self):
        J = 0
        for i in range(self.vocab_size):
            for j in range(self.vocab_size):
                if self.co_occurrence_matrix[i][j] > 0:
                    x_ij = self.co_occurrence_matrix[i][j]
                    weight = self.weight_function(x_ij)
                    prediction = self.dot_product(self.word_vectors[i], self.word_vectors[j]) + self.biases[i] + self.biases[j]
                    J += weight * (prediction - math.log(x_ij)) ** 2
        return J

    def weight_function(self, x_ij):
        if x_ij < self.x_max:
            return (x_ij / self.x_max) ** self.alpha
        else:
            return 1

    def dot_product(self, vec_a, vec_b):
        return sum(a * b for a, b in zip(vec_a, vec_b))

    def train(self, tokens, window_size, epochs):
        self.build_vocab(tokens)
        self.build_co_occurrence_matrix(tokens, window_size)
        
        for epoch in range(epochs):
            print(f"Epoch: {epoch}")
            for i in range(self.vocab_size):
                for j in range(self.vocab_size):
                    if self.co_occurrence_matrix[i][j] > 0:
                        x_ij = self.co_occurrence_matrix[i][j]
                        weight = self.weight_function(x_ij)
                        prediction = self.dot_product(self.word_vectors[i], self.word_vectors[j]) + self.biases[i] + self.biases[j]
                        
                        error = prediction - math.log(x_ij)
                        
                        for k in range(self.vector_dim):
                            grad = weight * error * self.word_vectors[j][k]
                            self.word_vectors[i][k] -= self.learning_rate * grad
                            
                        # Actualizar sesgos
                        self.biases[i] -= self.learning_rate * weight * error
                        self.biases[j] -= self.learning_rate * weight * error

    def save_embeddings(self, file_path):
        with open(file_path, 'w') as f:
            for i in range(self.vocab_size):
                f.write(f"{self.index_to_word[i]} {' '.join(map(str, self.word_vectors[i]))}\n")
