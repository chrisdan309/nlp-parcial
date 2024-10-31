import math
import random

class LSA:
    def __init__(self, documents, k, max_iterations=100, tolerance=1e-3):
        self.documents = documents
        self.k = k
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.terms = []
        self.X = []
        self.U = []
        self.Sigma = []
        self.Vt = []
        self.X_k = []

    def build_term_document_matrix(self):
        terms = {}
        for document in self.documents:
            for word in document.split():
                if word not in terms:
                    terms[word] = len(terms)

        X = [[0] * len(terms) for _ in range(len(self.documents))]
        for i, document in enumerate(self.documents):
            for word in document.split():
                X[i][terms[word]] += 1

        for j in range(len(terms)):
            df = sum(1 for i in range(len(self.documents)) if X[i][j] > 0)
            idf = math.log(len(self.documents) / (df + 1))
            for i in range(len(self.documents)):
                tf = X[i][j] / (sum(X[i]) + 1)
                X[i][j] = tf * idf

        self.X = X
        self.terms = list(terms.keys())

    def power_method(self, matrix):
        m, n = len(matrix), len(matrix[0])
        b_k = [random.random() for _ in range(n)]
        
        norm_b_k = math.sqrt(sum(x**2 for x in b_k))
        b_k = [x / norm_b_k for x in b_k]
        
        for _ in range(self.max_iterations):
            b_k1 = [sum(matrix[i][j] * b_k[j] for j in range(n)) for i in range(m)]
            b_k1 = [sum(matrix[j][i] * b_k1[j] for j in range(m)) for i in range(n)]
            
            norm_b_k1 = math.sqrt(sum(x**2 for x in b_k1))
            b_k1 = [x / norm_b_k1 for x in b_k1]

            if all(abs(b_k[i] - b_k1[i]) < self.tolerance for i in range(n)):
                return norm_b_k1, b_k1
            
            b_k = b_k1
        return norm_b_k1, b_k1

    def approximate_svd(self):
        A = [row[:] for row in self.X]
        U, Sigma, Vt = [], [], []

        for _ in range(self.k):
            singular_value, v = self.power_method(A)
            u = [sum(A[i][j] * v[j] for j in range(len(v))) / singular_value for i in range(len(A))]

            U.append(u)
            Sigma.append(singular_value)
            Vt.append(v)

            for i in range(len(A)):
                for j in range(len(A[0])):
                    A[i][j] -= singular_value * u[i] * v[j]

        self.U = [list(col) for col in zip(*U)]
        self.Sigma = [[Sigma[i] if i == j else 0 for j in range(self.k)] for i in range(self.k)]
        self.Vt = Vt

    def reduce_dimensionality(self):
        self.X_k = []

        for i in range(len(self.U)):
            row = []
            for j in range(len(self.Vt[0])):
                value = 0
                for p in range(self.k):
                    value += self.U[i][p] * self.Sigma[p][p] * self.Vt[p][j]                
                row.append(value)
            self.X_k.append(row)


    def calculate_similarity(self):
        def cosine_similarity(v1, v2):
            dot_product = sum(v1[i] * v2[i] for i in range(len(v1)))
            magnitude_v1 = math.sqrt(sum(x**2 for x in v1))
            magnitude_v2 = math.sqrt(sum(x**2 for x in v2))
            if magnitude_v1 == 0 or magnitude_v2 == 0:
                return 0
            return dot_product / (magnitude_v1 * magnitude_v2)

        return [[cosine_similarity(self.X_k[i], self.X_k[j]) for j in range(len(self.X_k))] for i in range(len(self.X_k))]
