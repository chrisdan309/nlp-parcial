import random
import math

class Word2Vec:
    def __init__(self, vocab, embedding_dim=100, window_size=2, negative_samples=5, learning_rate=0.01, epochs=10, sg=1):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.negative_samples = negative_samples
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.sg = sg
        self.W_in, self.W_out = self._initialize_vectors()

    def _initialize_vectors(self):
        W_in = [[random.uniform(-0.5, 0.5) for _ in range(self.embedding_dim)] for _ in range(self.vocab_size)]
        W_out = [[random.uniform(-0.5, 0.5) for _ in range(self.embedding_dim)] for _ in range(self.vocab_size)]
        return W_in, W_out

    def _sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def _dot_product(self, vec1, vec2):
        return sum(x * y for x, y in zip(vec1, vec2))

    def _scalar_multiply(self, vec, scalar):
        return [scalar * x for x in vec]

    def _vector_add(self, vec1, vec2):
        return [x + y for x, y in zip(vec1, vec2)]

    def _get_context(self, tokens, idx):
        start = max(0, idx - self.window_size)
        end = min(len(tokens), idx + self.window_size + 1)
        return [tokens[i] for i in range(start, end) if i != idx]

    def _negative_sampling(self, target_idx):
        negative_samples = []
        while len(negative_samples) < self.negative_samples:
            sample = random.randint(0, self.vocab_size - 1)
            if sample != target_idx:
                negative_samples.append(sample)
        return negative_samples

    def cbow_loss(self, context_vector, target_vector):
        """
        Función de pérdida para CBOW:
        Calcula la pérdida para una palabra objetivo dada un vector de contexto promedio.
        """
        dot_product = self._dot_product(context_vector, target_vector)
        prediction = self._sigmoid(dot_product)
        loss = -math.log(prediction)
        gradient = prediction - 1  # Gradiente para el descenso de gradiente (label=1 para CBOW positivo)
        
        return loss, gradient

    def skipgram_loss(self, target_vector, context_vector, label):
        """
        Función de pérdida para Skip-Gram:
        Calcula la pérdida para un par de palabras (target, context) dado el label (1 para positivo, 0 para negativo).
        """
        dot_product = self._dot_product(target_vector, context_vector)
        prediction = self._sigmoid(dot_product)
        loss = -math.log(prediction) if label == 1 else -math.log(1 - prediction)
        gradient = prediction - label  # Gradiente basado en el label
        
        return loss, gradient

    def train(self, tokens):
        token_indices = [self.vocab[token] for token in tokens if token in self.vocab]

        for epoch in range(self.epochs):
            total_loss = 0
            for idx, target_idx in enumerate(token_indices):
                context_words = self._get_context(token_indices, idx)
                
                if self.sg == 1:
                    # Skip-Gram: La palabra objetivo predice las palabras de contexto
                    for context_word_idx in context_words:
                        total_loss += self._skipgram_step(target_idx, context_word_idx)
                else:
                    # CBOW: Las palabras de contexto predicen la palabra objetivo
                    total_loss += self._cbow_step(context_words, target_idx)
            
            print(f"Epoch {epoch + 1}, Loss: {total_loss}")

    def _skipgram_step(self, target_idx, context_word_idx):
        # Calcular pérdida y gradiente para el par positivo
        pos_loss, pos_gradient = self.skipgram_loss(self.W_in[target_idx], self.W_out[context_word_idx], 1)
        
        # Actualización para el par positivo
        self.W_in[target_idx] = self._vector_add(self.W_in[target_idx], 
                                                 self._scalar_multiply(self.W_out[context_word_idx], -self.learning_rate * pos_gradient))
        self.W_out[context_word_idx] = self._vector_add(self.W_out[context_word_idx], 
                                                        self._scalar_multiply(self.W_in[target_idx], -self.learning_rate * pos_gradient))

        # Actualización para los pares negativos
        neg_samples = self._negative_sampling(target_idx)
        neg_loss = 0
        for neg_word_idx in neg_samples:
            neg_loss_sample, neg_gradient = self.skipgram_loss(self.W_in[target_idx], self.W_out[neg_word_idx], 0)
            neg_loss += neg_loss_sample

            self.W_in[target_idx] = self._vector_add(self.W_in[target_idx], 
                                                     self._scalar_multiply(self.W_out[neg_word_idx], -self.learning_rate * neg_gradient))
            self.W_out[neg_word_idx] = self._vector_add(self.W_out[neg_word_idx], 
                                                        self._scalar_multiply(self.W_in[target_idx], -self.learning_rate * neg_gradient))

        return pos_loss + neg_loss

    def _cbow_step(self, context_words, target_idx):
        # Calcular el vector promedio del contexto
        context_vector = [0] * self.embedding_dim
        for context_word_idx in context_words:
            context_vector = self._vector_add(context_vector, self.W_in[context_word_idx])
        context_vector = self._scalar_multiply(context_vector, 1 / len(context_words))

        # Calcular pérdida y gradiente para el par positivo
        pos_loss, pos_gradient = self.cbow_loss(context_vector, self.W_out[target_idx])
        
        # Actualización para el par positivo
        self.W_out[target_idx] = self._vector_add(self.W_out[target_idx], 
                                                  self._scalar_multiply(context_vector, -self.learning_rate * pos_gradient))
        
        for context_word_idx in context_words:
            self.W_in[context_word_idx] = self._vector_add(self.W_in[context_word_idx], 
                                                           self._scalar_multiply(self.W_out[target_idx], -self.learning_rate * pos_gradient))

        # Actualización para los pares negativos
        neg_samples = self._negative_sampling(target_idx)
        neg_loss = 0
        for neg_word_idx in neg_samples:
            neg_loss_sample, neg_gradient = self.skipgram_loss(context_vector, self.W_out[neg_word_idx], 0)
            neg_loss += neg_loss_sample

            self.W_out[neg_word_idx] = self._vector_add(self.W_out[neg_word_idx], 
                                                        self._scalar_multiply(context_vector, -self.learning_rate * neg_gradient))

            for context_word_idx in context_words:
                self.W_in[context_word_idx] = self._vector_add(self.W_in[context_word_idx], 
                                                               self._scalar_multiply(self.W_out[neg_word_idx], -self.learning_rate * neg_gradient))

        return pos_loss + neg_loss

    def get_embedding(self, word):
        word_idx = self.vocab.get(word)
        if word_idx is not None:
            return self.W_in[word_idx]
        else:
            return None

if __name__ == "__main__":
    tokens = ["el", "hombre", "en", "el", "parque", "vio", "otro", "hombre"]

    vocab = {word: idx for idx, word in enumerate(set(tokens))}

    # Creación y entrenamiento del modelo
    model = Word2Vec(vocab=vocab, embedding_dim=10, window_size=2, negative_samples=5, learning_rate=0.01, epochs=10)
    model.train(tokens)

    # Verificar que el embedding sea el mismo
    embedding = model.get_embedding("hombre")
    print("Embedding de la palabra 'hombre':", embedding)
