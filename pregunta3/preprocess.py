import re

class TextProcessor:
    def __init__(self, corpus_path):
        self.corpus_path = corpus_path
        self.stopwords = ['a', 'y', 'de', 'la', 'el', 'con', 'un', 'como', 'que', 'por', 'en', 'o', 'del',
                        'lo', 'para', 'ha', 'lo', 'se', 'al', 'e', 'una', 'su', 'entre', '', 'm', 'n', 'desde',
                        'i', 'pero', 'no', 'ya', 'sobre', 'si']
        self.suffixes = [
            'amiento', 'imientos', 'ación', 'aciones', 'adora', 'adoras', 'ador', 'adores', 
            'ante', 'antes', 'ancia', 'ancias', 'adora', 'adoras', 'ación', 'aciones',
            'imiento', 'imientos', 'ico', 'ica', 'icos', 'icas', 'iva', 'ivo', 'ivas', 'ivos',
            'mente', 'idad', 'idades', 'iva', 'ivo', 'ivas', 'ivos', 'anza', 'anzas', 'ero', 'era', 'eros', 'eras',
            'ces', 's', 'es'
        ]

    def tokenize(self, corpus):
        tokens = []
        for line in corpus:
            tokens += re.findall(r'\b[a-zA-Zñáéíóúü]+\b', line)
        return tokens

    def lematizacion(self, tokens):
        new_tokens = []
        for token in tokens:
            original_token = token
            for suffix in sorted(self.suffixes, key=len, reverse=True):
                if token.endswith(suffix):
                    token = token[:-len(suffix)]
                    break
            new_tokens.append(token if len(token) > 0 else original_token)
        return new_tokens

    def remove_stopwords(self, tokens):
        new_tokens = [token for token in tokens if token not in self.stopwords]
        return new_tokens

    def filter_tokens(self, tokens, min_frequency):
        token_counts = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
        return [token for token in tokens if token_counts[token] > min_frequency]

    def preprocess(self, corpus, min_frequency=5):
        tokens = self.tokenize(corpus)
        tokens = self.lematizacion(tokens)
        tokens = self.remove_stopwords(tokens)
        tokens = self.filter_tokens(tokens, min_frequency)
        return tokens
    
    
    def preprocess_by_batches(self, batch_size, top, min_frequency=5):
        token_counts = {}
        ordered_tokens = []

        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            batch = []
            cont = 0
            for line in f:
                batch.append(line.lower())
                if len(batch) == batch_size:
                    tokens = self.tokenize(batch)
                    tokens = self.lematizacion(tokens)
                    tokens = self.remove_stopwords(tokens)
                    
                    ordered_tokens.extend(tokens)

                    for token in tokens:
                        token_counts[token] = token_counts.get(token, 0) + 1

                    batch = []
                    cont += batch_size
                    print(f"proceso: {cont} lineas")

                if cont >= batch_size * top:
                    break

            if batch:
                tokens = self.tokenize(batch)
                tokens = self.lematizacion(tokens)
                tokens = self.remove_stopwords(tokens)

                ordered_tokens.extend(tokens)

                for token in tokens:
                    token_counts[token] = token_counts.get(token, 0) + 1

                cont += len(batch)
                print(f"proceso: {cont} lineas")

        filtered_tokens = [token for token in ordered_tokens if token_counts.get(token, 0) > min_frequency]
        
        return filtered_tokens
