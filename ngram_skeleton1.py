import math
import random
import datetime
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay


################################################################################
# Part 0: Utility Functions
################################################################################

COUNTRY_CODES = ['af', 'cn', 'de', 'fi', 'fr', 'in', 'ir', 'pk', 'za']


def start_pad(n):
    """ Returns a padding string of length n to append to the front of text
        as a pre-processing step to building n-grams """
    return '~' * n


def ngrams(n, text):
    text = start_pad(n) + text  # başa ~ ekle
    output = []
    for i in range(n, len(text)):
        context = text[i - n:i]
        char = text[i]
        output.append((context, char))
    return output


def create_ngram_model(model_class, path, n=2, k=0):
    """ Creates and returns a new n-gram model trained on the city names
        found in the path file """
    model = model_class(n, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            city = line.strip()
            model.update(city)
    return model


def create_ngram_model_lines(model_class, path, n=2, k=0):
    """ Creates and returns a new n-gram model trained on the city names
        found in the path file (each line may have multiple names) """
    model = model_class(n, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            cities = line.strip().split()
            for city in cities:
                model.update(city)
    return model

def count_multiple_char_group(word, n):
    letter_group = [word[i:i+n] for i in range(len(word)-n-1)]
    return {k: letter_group.count(k) for k in set(letter_group)}


def load_dataset(folder):
    x = []
    y = []
    for code in COUNTRY_CODES:
        with open("./" + folder + "/" + code + ".txt", encoding='utf-8', errors='ignore') as input_file:
            for city in input_file:
                x.append(city.strip())
                y.append(code)
    return x, y


################################################################################
# Part 1: Basic N-Gram Model
################################################################################

class NgramModel(object):
    """ A basic n-gram model using add-k smoothing """

    def __init__(self, n, k):
        self.smoothing = k
        self.n = n
        self.vocab = list()
        self.ngrams = dict()
        self.base_chars = set("abcdefghijklmnopqrstuvwxyz '-^$")

    def get_vocab(self):
        """ Returns the set of characters in the vocab """
        return set(self.vocab)

    def update(self, text):
        """ Updates the model n-grams based on text """
        filtered_text = ''.join(c for c in text.lower() if c in self.base_chars)
        n_grams = ngrams(self.n, filtered_text)
        for context, char in n_grams:
            if char not in self.vocab:
                self.vocab.append(char)
            if context not in self.ngrams:
                self.ngrams[context] = dict()
            if char not in self.ngrams[context]:
                self.ngrams[context][char] = 0
            self.ngrams[context][char] += 1

    def prob(self, context, char):
        """ Returns the probability of char appearing after context """
        V = len(self.vocab)
        if context in self.ngrams:
            context_dict = self.ngrams[context]
            context_count = sum(context_dict.values())
            char_count = context_dict.get(char, 0)
            return (char_count + self.smoothing) / (context_count + self.smoothing * V)
        else:
            return 1 / V if V > 0 else 0

    def random_char(self, context):
        """ Returns a random character based on the given context and the
            n-grams learned by this model """
        r = random.random()
        total = 0.0
        vocab_sorted = sorted(self.get_vocab())
        for char in vocab_sorted:
            total += self.prob(context, char)
            if r < total:
                return char
        return vocab_sorted[-1]  # fallback

    def random_text(self, length):
        """ Returns text of the specified character length based on the
            n-grams learned by this model """
        context = start_pad(self.n)
        result = ''
        for _ in range(length):
            next_char = self.random_char(context[-self.n:])
            result += next_char
            context += next_char
        return result

    def perplexity(self, text):
        """ Returns the perplexity of text based on the n-grams learned by
            this model """
        n_grams = ngrams(self.n, text)
        log_prob_sum = 0.0
        for context, char in n_grams:
            prob_val = self.prob(context, char)
            if prob_val > 0:
                log_prob_sum += math.log(prob_val)
            else:
                return float('inf')  # if prob is zero, perplexity is infinite
        l = len(n_grams)
        if l == 0:
            return float('inf')
        avg_log_prob = log_prob_sum / l
        return math.exp(-avg_log_prob)

################################################################################
# Part 2: N-Gram Model with Interpolation
################################################################################

class NgramModelWithInterpolation(NgramModel):
    """ An n-gram model with interpolation """

    def __init__(self, n, k):
        super(NgramModelWithInterpolation, self).__init__(n, k)
        self.n_grams_all = []
        for i in range(0, n + 1):
            self.n_grams_all.append(NgramModel(i, k))
        self.lambdas = None 
    
    def get_vocab(self):
        return set(self.n_grams_all[-1].get_vocab())
   

    def update(self, text):
        for i in range(0, self.n + 1):
            self.n_grams_all[i].update(text)

    def prob(self, context, char):
        # Lambda değerlerini al (eğer ayarlanmamışsa varsayılan değerler kullanılır)
        lambdas = self.set_lambdas()
        
        total_prob = 0.0
        for i in range(len(lambdas)):
            # i, kullanılacak n-gram boyutunu temsil eder (0=unigram, 1=bigram, ...)
            if i <= len(context):
                # Uygun boyutta context al
                sub_context = context[-i:] if i > 0 else ''
                
                # Alt modelin olasılığını hesapla ve lambda ağırlığıyla çarp
                prob = self.n_grams_all[i].prob(sub_context, char)
                total_prob += lambdas[i] * prob
        
        return total_prob

    def set_lambdas(self, lambdas=None):
        if lambdas is not None:
            # Verilen lambda değerlerini normalleştir (toplamı 1 yap)
            total = sum(lambdas)
            self.lambdas = [l/total for l in lambdas]
        else:
            # Varsayılan değerler: daha yüksek n-gram'lara daha fazla ağırlık ver
            # Örneğin n=3 için: [0.1, 0.2, 0.3, 0.4]
            self.lambdas = [0.1]  # Unigram
            if self.n >= 1:
                self.lambdas.extend([0.2 * (i+1) for i in range(self.n)])
            total = sum(self.lambdas)
            self.lambdas = [l/total for l in self.lambdas]
        
        return self.lambdas
    
    def update_lambdas(self, lambdas):
        """Lambda değerlerini günceller"""
        total = sum(lambdas)
        self.lambdas = [l/total for l in lambdas]  # Normalize edilmiş lambda değerleri

################################################################################
# Part 3: Your N-Gram Model Experimentation
################################################################################


class CountriesModel:

    def __init__(self, n, k, interpolation=False):
        self.models = {}
        self.n = n
        for code in COUNTRY_CODES:
            if interpolation:
                self.models[code] = (
                    create_ngram_model(NgramModelWithInterpolation, "data/train/" + code + ".txt", n=n, k=k))
            else:
                self.models[code] = (create_ngram_model(NgramModel, "data/train/" + code + ".txt", n=n, k=k))

    def predict_country(self, city):
        max_prob = -float('inf')  # Log olasılık kullanacağımız için -sonsuz başlatıyoruz
        arg_max = ""
        
        # Şehir ismini işle (başlangıç/bitiş tokenları ekle)
        processed_city = '^' + ''.join(c for c in city.strip().lower() if c.isalpha() or c in " '-") + '$'
        
        for country_code, model in self.models.items():
            try:
                # Şehrin log olasılığını hesapla
                log_prob = 0.0
                contexts = ngrams(model.n, processed_city)
                if not contexts:  # Boş şehir isimlerini atla
                    continue
                for context, char in contexts:
                    p = model.prob(context, char)
                    if p <= 1e-10:
                        log_prob = -float('inf')
                        break
                    log_prob += math.log(p)
                
                normalized_log_prob = log_prob / len(processed_city) if len(processed_city) > 0 else log_prob

                # En yüksek olasılıklı ülkeyi güncelle
                if normalized_log_prob > max_prob:
                    max_prob = normalized_log_prob
                    arg_max = country_code
                    
            except Exception as e:
                print(f"Hata: {country_code} modelinde {city} işlenirken - {str(e)}")
                continue
        
        # Eğer hiçbir model uygun sonuç vermediyse rastgele bir ülke seç
        return arg_max if arg_max else random.choice(COUNTRY_CODES)

    def fit(self, cities):
        return [self.predict_country(cities[i]) for i in range(len(cities))]

    def update_lambdas(self, lambdas):
        for code in self.models:
            if hasattr(self.models[code], 'update_lambdas'):  # Metodun varlığını kontrol et
                self.models[code].update_lambdas(lambdas)
            else:
                print(f"Uyarı: {code} modelinde update_lambdas metodu bulunamadı")


if __name__ == '__main__':

    print(f'NLP Homework | {datetime.datetime.now()}')
    print(f'##########################################################################################################')

    shakespeare_n = 2

    print(f'Generating {shakespeare_n}-gram shakespeare')
    print(f'##########################################################################################################')
    m = create_ngram_model(NgramModel, 'data/shakespeare_input.txt', shakespeare_n)
    print(m.random_text(250))

    print(f'##########################################################################################################')

    shakespeare_n = 3
    print(f'Generating {shakespeare_n}-gram shakespeare')
    print(f'##########################################################################################################')
    m = create_ngram_model(NgramModel, 'data/shakespeare_input.txt', shakespeare_n)
    print(m.random_text(250))

    print(f'##########################################################################################################')

    shakespeare_n = 4
    print(f'Generating {shakespeare_n}-gram shakespeare')
    print(f'##########################################################################################################')
    m = create_ngram_model(NgramModel, 'data/shakespeare_input.txt', shakespeare_n)
    print(m.random_text(250))

    print(f'##########################################################################################################')

    shakespeare_n = 7
    print(f'Generating {shakespeare_n}-gram shakespeare')
    print(f'##########################################################################################################')
    m = create_ngram_model(NgramModel, 'data/shakespeare_input.txt', shakespeare_n)
    print(m.random_text(250))

    print(f'##########################################################################################################')

    # war_peace_n = 2

    # print(f'Generating {war_peace_n }-gram war peace')
    # print(f'##########################################################################################################')
    # m = create_ngram_model(NgramModel, 'warpeace_input.txt', war_peace_n)
    # print(m.random_text(250))

    # print(f'##########################################################################################################')

    # war_peace_n = 3
    # print(f'Generating {war_peace_n }-gram war peace')
    # print(f'##########################################################################################################')
    # m = create_ngram_model(NgramModel, 'warpeace_input.txt', war_peace_n)
    # print(m.random_text(250))

    # print(f'##########################################################################################################')

    # war_peace_n = 4
    # print(f'Generating {war_peace_n}-gram war peace')
    # print(f'##########################################################################################################')
    # m = create_ngram_model(NgramModel, 'warpeace_input.txt', war_peace_n)
    # print(m.random_text(250))

    # print(f'##########################################################################################################')

    # war_peace_n = 7
    # print(f'Generating {war_peace_n}-gram war peace')
    # print(f'##########################################################################################################')
    # m = create_ngram_model(NgramModel, 'warpeace_input.txt', war_peace_n)
    # print(m.random_text(250))

    print(f'##########################################################################################################')

    x_train, y_train = load_dataset('data/train')
    x_dev, y_dev = load_dataset('data/val')

    n, k = (3,) * 2

    print(f'##########################################################################################################')
    print(f'n: {n}, k: {k}')
    print(f'##########################################################################################################')

    model = CountriesModel(n, k, interpolation=True)

    lambdas = (0.1, 0.5, 0.4)
    model.update_lambdas(lambdas)

    y_train_pred = model.fit(x_train)
    f_train = f1_score(y_train, y_train_pred, average='micro')
    print(f'Performance scores for train | Precision: {f_train}')

    print(f'##########################################################################################################')

    y_dev_pred = model.fit(x_dev)
    f_dev = f1_score(y_dev, y_dev_pred, average='micro')
    print(f'Performance scores for development | Precision: {f_dev}')

    with open("data/cities_test.txt", encoding='utf-8', errors='ignore') as file:
        x_test = [line.strip() for line in file]

    y_test_pred = model.fit(x_test)

    with open('data/test_labels.txt', 'wt', encoding='utf8') as file:
        for p in y_test_pred:
            file.write(str(p) + '\n')
