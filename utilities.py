import time
import numpy as np
from scipy.stats import multinomial


class Corpus:
    __corpus = None
    __number_of_titles = None
    __documents = None
    __vocabulary = None
    __number_of_documents = None
    __numbered_documents = None
    __punctuations = None

    def __init__(self, corpus, number_of_titles):
        self.__corpus = corpus
        self.__number_of_titles = number_of_titles

        self.__documents = []
        self.__numbered_documents = []

        self.__punctuations = ['.', ',', '!', '?', ':', ';', '"', '\'']

        self.extract_documents()
        self.make_vocabulary()
        self.turn_documents_to_numbers()

    def extract_documents(self):
        print('[*] Extracting Documents Out of the Corpus')
        documents = self.__corpus.split('<DOC>')

        for document in documents:
            doc_id = document.split('<DOCNO>')[-1].split('</DOCNO>')[0].strip()
            doc_text = document.split('<TEXT>')[-1].split('</TEXT>')[0].strip()

            if doc_id != '' and doc_text != '':
                self.__documents.append({'doc_id': doc_id, 'doc_text': doc_text})

        self.__number_of_documents = len(self.__documents)

    def get_documents(self):
        return self.__documents

    def get_document_by_id(self, doc_id):
        for document in self.__documents:
            if document['doc_id'] == doc_id:
                return document

        return None

    def get_number_of_titles(self):
        return self.__number_of_titles

    def get_number_of_documents(self):
        return self.__number_of_documents

    def turn_documents_to_numbers(self):
        print('[*] Turning Documents to Numerical Arrays')
        index = 1
        for document in self.__documents:
            print(f'\t[*] Analaysing the {index}/{self.__number_of_documents} document')
            document_text = document['doc_text']
            words = document_text.strip().split(' ')
            doc_bag_of_words = []
            for word in words:
                if word not in self.__punctuations:
                    doc_bag_of_words.append(self.__vocabulary.index(word.lower()))

            self.__numbered_documents.append(doc_bag_of_words)
            index += 1

    def get_numbered_documents(self):
        return self.__numbered_documents

    def make_vocabulary(self):
        print('[*] Making Vocabulary From the Corpus')
        vocabulary = set()

        for document in self.__documents:
            words = document['doc_text'].strip().split(' ')
            for word in words:
                if word not in self.__punctuations:
                    vocabulary.add(word.lower())

        self.__vocabulary = list(vocabulary)

    def get_number_of_vocabularies(self):
        return len(self.__vocabulary)


class LDA:
    __corpus = None
    __alpha = None
    __beta = None
    __number_of_titles = None
    __word_topic_count = None
    __document_topic_count = None
    __theta_matrix = None
    __phi_matrix = None
    __number_of_vocabs = None
    __numbered_documents = None
    __document_word_title_assignment = None
    __number_of_documents = None
    __distribution = None

    def __init__(self, corpus, alpha, beta):
        self.__corpus = corpus
        self.__alpha = alpha
        self.__beta = beta

        self.__number_of_titles = corpus.get_number_of_titles()
        self.__number_of_vocabs = corpus.get_number_of_vocabularies()
        self.__numbered_documents = corpus.get_numbered_documents()
        self.__number_of_documents = len(self.__numbered_documents)

        self.__word_topic_count = np.zeros((self.__number_of_titles, self.__number_of_vocabs))
        self.__document_word_title_assignment = []
        self.__document_topic_count = np.zeros((self.__number_of_documents, self.__number_of_titles))

        self.assign_initial_topics_to_words()
        self.compute_word_topic_count()
        self.compute_document_topic_count()

    def assign_initial_topics_to_words(self):
        print('[*] Assigning Initial Topics to the Words of Each Document')
        for i in range(self.__number_of_documents):
            print(f'\tWorking on Document {i}/{self.__number_of_documents}...')
            initial_titles = np.random.randint(0, self.__number_of_titles, (len(self.__numbered_documents[i], )))
            self.__document_word_title_assignment.append(initial_titles)

    def compute_word_topic_count(self):
        print('[*] Computing the Word-Topic Matrix (The Number of Words Assigned to Each Topic)')
        for i in range(self.__number_of_documents):
            print(f'\tWorking on Document {i}/{self.__number_of_documents}...')
            for j in range(len(self.__numbered_documents[i])):
                index1 = self.__document_word_title_assignment[i][j]
                index2 = self.__numbered_documents[i][j]
                self.__word_topic_count[index1, index2] += 1

    def compute_document_topic_count(self):
        print('[*] Computing the Document-Topic Matrix (The Distribution of Each Topic in Each Document')
        for i in range(self.__number_of_documents):
            print(f'\tWorking on Document {i}/{self.__number_of_documents}...')
            for j in range(len(self.__numbered_documents[i])):
                index1 = i
                index2 = self.__document_word_title_assignment[i][j]
                self.__document_topic_count[index1, index2] += 1

    def train(self, max_iter=5000):
        s_time = time.time()
        print('[*] Training Started...')
        for iteration in range(max_iter):
            print('\t Iteration : ', iteration)
            for i in range(self.__number_of_documents):
                for j in range(len(self.__numbered_documents[i])):
                    initial_topic = self.__document_word_title_assignment[i][j]
                    word_id = self.__numbered_documents[i][j]

                    self.__document_topic_count[i, initial_topic] -= 1
                    self.__word_topic_count[initial_topic, word_id] -= 1

                    denominator_a = sum(self.__document_topic_count[i, :]) + self.__number_of_titles * self.__alpha
                    denominator_b = np.sum(self.__word_topic_count, axis=1) + self.__number_of_vocabs * self.__beta

                    p_z = ((self.__word_topic_count[:, word_id] + self.__beta)/denominator_b) *\
                          ((self.__document_topic_count[i, :] + self.__alpha)/denominator_a)
                    self.__distribution = p_z

                    new_topic = list(multinomial.rvs(1, p=p_z/sum(p_z))).index(1)
                    self.__document_word_title_assignment[i][j] = new_topic
                    self.__document_topic_count[i, new_topic] += 1
                    self.__word_topic_count[new_topic, word_id] += 1

                    # if initial_topic != new_topic:
                    #     print(f'[*] Document : {i+1}, Word : {j}, Topic : {initial_topic} => {new_topic}')
        f_time = time.time()
        print('[*] Training time is : ', f_time-s_time)

    def get_theta(self):
        self.__theta_matrix = np.zeros(self.__document_topic_count.shape)

        for col in range(self.__theta_matrix.shape[1]):
            self.__theta_matrix[:, col] = (self.__document_topic_count + self.__alpha)[:, col] / \
                                          (np.sum(self.__document_topic_count + self.__alpha, axis=1))

        return self.__theta_matrix

    def get_phi(self):
        self.__phi_matrix = np.zeros(self.__word_topic_count.shape)

        for col in range(self.__phi_matrix.shape[1]):
            self.__phi_matrix[:, col] = (self.__word_topic_count + self.__beta)[:, col] / \
                                        (np.sum(self.__word_topic_count + self.__beta, axis=1))

        return self.__phi_matrix

    def perplexity(self):
        perp = 0
        for prob in self.__distribution:
            probability = prob / sum(self.__distribution)

            perp += probability * np.log2(probability)

        perp = 2 ** (-perp)

        return perp

