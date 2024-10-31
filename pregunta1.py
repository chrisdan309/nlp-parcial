# Código inspirado en Modelos-lenguaje2.ipynb

def tokenize(text):
    return text.split()

class bigram():
    corpus = None
    vocab = None
    bigram_count = {}
    unigram_count = {}
    
    def __init__(self, corpus:list, vocab):
        self.corpus = corpus
        self.vocab = vocab
        for word in vocab:
            cont = 0
            for sentence in corpus:
                sentence_token = tokenize(sentence)
                if word in sentence_token:
                    cont += 1

            self.unigram_count[word] = cont
        self.unigram_count["<s>"] = len(corpus)
        self.unigram_count["</s>"] = len(corpus)

    def train(self):
        for sentence in corpus:
            sentence_token = tokenize(sentence)

            if ('<s>',sentence_token[0]) not in self.bigram_count.keys():
                    self.bigram_count[('<s>',sentence_token[0])] = 1
            else:
                self.bigram_count[('<s>',sentence_token[0])] += 1
            
            for i in range(len(sentence_token)-1):
                word_1 = sentence_token[i]
                word_2 = sentence_token[i+1]
                if (word_1, word_2) not in self.bigram_count.keys():
                    self.bigram_count[(word_1, word_2)] = 1
                else:
                    self.bigram_count[(word_1, word_2)] += 1

            if (sentence_token[-1],'</s>') not in self.bigram_count.keys():
                self.bigram_count[(sentence_token[-1],'</s>')] = 1
            else:
                self.bigram_count[(sentence_token[-1],'</s>')] += 1

    def calculate_probability(self,word1,word2):
        count_word1_word2 = self.bigram_count[(word1,word2)]
        if word1 in vocab:
            count_word1 = self.unigram_count[word1]
        else:
            return 0
        probability = count_word1_word2/count_word1
        return probability
        

    def calculate_probability_add_k_smoothing(self,word1,word2,k):
        v = len(vocab)
        count_word1_word2 = k
        if (word1,word2) in self.bigram_count.keys():
            count_word1_word2 = self.bigram_count[(word1,word2)] + k
        
        count_word1 = k*v
        if word1 in self.vocab:
            count_word1 = self.unigram_count[word1] + k*v
        
        probability = count_word1_word2/count_word1
        return probability
        

    def backoff(self, word1, word2,lambda_factor = 1):
        if (word1,word2) in self.bigram_count.keys():
            return self.calculate_probability(word1,word2)
        else:
            return lambda_factor*self.unigram_count[word2]/len(vocab)

if __name__ == "__main__":    
    corpus = ["all models are wrong",
                "a model is wrong",
                "some models are useful"]

    vocab = ['<s>','</s>','a','all','are','model','models','some','useful','wrong']
    print("Corpus")
    print(corpus)
    bigrama = bigram(corpus, vocab)
    bigrama.train()
    print("\nunigramas:")
    print(bigrama.unigram_count)
    print("\nbigramas:")
    print(bigrama.bigram_count)

    print("\nProbabilities")
    for tupla in bigrama.bigram_count.keys():
        probability = bigrama.calculate_probability(tupla[0],tupla[1])
        print("P(",tupla[1],"|",tupla[0],")=",probability)

    print("\nadd-one smoothing")
    k=1.
    for tupla in bigrama.bigram_count.keys():
        probability = bigrama.calculate_probability_add_k_smoothing(tupla[0],tupla[1],k)
        print("P(",tupla[1],"|",tupla[0],")=",probability)
    probability = bigrama.calculate_probability_add_k_smoothing("a","models",k)
    print("P(a|models)=",probability)

    print("\nadd-k smoothing, k=0.05")
    k=0.05
    for tupla in bigrama.bigram_count.keys():
        probability = bigrama.calculate_probability_add_k_smoothing(tupla[0],tupla[1],k)
        print("P(",tupla[1],"|",tupla[0],")=",probability)
    probability = bigrama.calculate_probability_add_k_smoothing("a","models",k)
    print("P(a|models)=",probability)

    print("\nadd-k smoothing, k=0.15")
    k=0.15
    for tupla in bigrama.bigram_count.keys():
        probability = bigrama.calculate_probability_add_k_smoothing(tupla[0],tupla[1],k)
        print("P(",tupla[1],"|",tupla[0],")=",probability)
    probability = bigrama.calculate_probability_add_k_smoothing("a","models",k)
    print("P(a|models)=",probability)


    print("\nbackoff")
    for tupla in bigrama.bigram_count.keys():
        probability = bigrama.backoff(tupla[0],tupla[1])
        print("P(",tupla[1],"|",tupla[0],")=",probability)
    probability = bigrama.backoff("a","models")
    print("P(a|models)=",probability)

    print("\nstupid backoff, lambda = 0.3")
    lambda_factor = 0.3
    for tupla in bigrama.bigram_count.keys():
        probability = bigrama.backoff(tupla[0],tupla[1],lambda_factor)
        print("P(",tupla[1],"|",tupla[0],")=",probability)
    probability = bigrama.backoff("a","models",k)
    print("P(a|models)=",probability)
