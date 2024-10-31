# Código inspirado en Modelos-lenguaje2.ipynb

def tokenize(text):
    return text.split()

class bigram():
    corpus = None
    vocab = None
    bigram_count = {}
    unigram_count = {}
    frequency_unigram_count = {}
    adjusted_counts = {}
    mle_probabilities = {}
    normalized_probabilities = {}
    def __init__(self, corpus:list, vocab: list):
        self.corpus = corpus
        self.vocab = vocab
        for word in vocab:
            cont = 0
            for sentence in corpus:
                sentence_token = tokenize(sentence)
                if word in sentence_token:
                    cont += 1

            self.unigram_count[word] = cont
        
        for sentence in corpus:
            sentence_token = tokenize(sentence)
            for word in sentence_token:
                if word not in self.vocab:
                    if '<UNK>' not in self.vocab:
                        self.vocab.append('<UNK>')
                        self.unigram_count['<UNK>'] = 1
                    else:
                        self.unigram_count['<UNK>'] += 1

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
                

    def good_turing(self):
        self.frequency_unigram_count[0] = 1
        unigramas = []
        for value in self.unigram_count.values():
            unigramas.append(value)

        unigramas = sorted(list(set(unigramas)))
        for key in unigramas:
            count = 0
            for value in self.unigram_count.values():
                if key == value:
                    count += 1
            self.frequency_unigram_count[key] = count
        N = sum(self.unigram_count.values())
        V = len(self.vocab)
        for word, count in self.unigram_count.items():
            adjusted_count = (count + 1) * (self.frequency_unigram_count.get(count + 1, 0) / self.frequency_unigram_count[count])
            self.adjusted_counts[word] = adjusted_count * (N / (N + V))

        print("Conteos ajustados (c^*):", self.adjusted_counts)

    def calculate_probability_adjusted(self, word):
        total_adjusted_count = sum(self.adjusted_counts.values())
        if word in self.adjusted_counts:
            return self.adjusted_counts[word] / total_adjusted_count
        else:
            return self.adjusted_counts['<UNK>'] / total_adjusted_count
        
    def calculate_mle_probability(self):
        total_count = sum(self.unigram_count.values())
        mle_probabilities = {}
        for word, count in self.unigram_count.items():
            if count == 3:
                mle_probabilities[word] = count / total_count
                print(f"P({word}) (MLE) = {mle_probabilities[word]:.6f}")
        return mle_probabilities
    
    def normalize_probabilities(self):
        total_adjusted_count = sum(self.adjusted_counts.values())
        print(f"Suma de probabilidades sin ajustar: {total_adjusted_count:.6f}")
        normalized_probabilities = {}
        for word, adjusted_count in self.adjusted_counts.items():
            normalized_probabilities[word] = adjusted_count / total_adjusted_count
        print("\nProbabilidades normalizadas:")
        for word, prob in normalized_probabilities.items():
            print(f"P({word}) = {prob:.6f}")
        print(f"Suma de probabilidades normalizadas: {sum(normalized_probabilities.values()):.6f}")
        return normalized_probabilities


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
    
    print("\nGood-Turing")
    bigrama.good_turing()

    print("\nMLE")
    bigrama.calculate_mle_probability()

    for word in vocab:
        probability = bigrama.calculate_probability_adjusted(word)
        print(f"P({word}) = {probability:.6f}")

    bigrama.normalize_probabilities()