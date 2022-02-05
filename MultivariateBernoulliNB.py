from aux_function import *
from decimal import *
import pickle
import math

class MultivariateBernoulliNB:

    def __init__(self, save=False, n=100):
        self.save = save
        self.n = n
        self.MB_avg_mutual_info = None
        self.u_words = None
        self.files = None
        self.filepath_list = None
        self.directories = None
        self.class_train = None
        self.class_test = None
        self.dictionary = None
        self.dictionary_test = None
        self.storage_accuracy = None
        self.all_word_list = None
        self.classes = None
        self.doc_train = None

        self.prepare_data(save)

        self.MB_features = None
        self.get_features()

        self.prior = None
        self.calc_prior()

        self.MB_p_of_w = None
        self.calc_prob_of_word()

        self.MB_p_of_d_given_c = None
        self.calc_p_of_d_given_c()

    def prepare_data(self, save=False):

        with open('directories', 'rb') as f:
            directories = pickle.load(f)
        with open('files', 'rb') as f:
            files = pickle.load(f)
        with open('filepath_list', 'rb') as f:
            filepath_list = pickle.load(f)
        with open('classes', 'rb') as f:
            classes = pickle.load(f)
        with open('doc_train', 'rb') as f:
            doc_train = pickle.load(f)

        with open('class_train', 'rb') as f:
            class_train = pickle.load(f)
        with open('class_test', 'rb') as f:
            class_test = pickle.load(f)
        with open('unique_words', "rb") as f:
            u_words = pickle.load(f)
        with open('dictionary', "rb") as f:
            dictionary = pickle.load(f)
        with open('MB_mutual_info_FS', 'rb') as f:
            MB_avg_mutual_info = pickle.load(f)
        with open('dictionary_test', 'rb') as f:
            dictionary_test = pickle.load(f)

        self.MB_avg_mutual_info = MB_avg_mutual_info
        self.u_words = u_words
        self.files = files
        self.filepath_list = filepath_list
        self.directories = directories
        self.class_train = class_train
        self.class_test = class_test
        self.dictionary = dictionary
        self.dictionary_test = dictionary_test
        self.classes = classes
        self.doc_train = doc_train

    def get_features(self):
        list = []
        tmp = np.matrix(self.MB_avg_mutual_info)

        for i in range(self.n):
            index = tmp.argmax()
            tmp[(index // len(self.u_words)), index - (index // len(self.u_words) * len(self.u_words))] = -1
            list.append(index - (index // len(self.u_words) * len(self.u_words)))
        self.MB_features = [self.u_words[list[i]] for i in range(len(list))]

    def calc_prior(self):
        prior = []
        for category_files in self.files:
            prior.append(len(category_files) / len(self.filepath_list))
        self.prior = prior

    def calc_prob_of_word(self):
        MB_p_of_w = np.zeros((len(self.directories), len(self.MB_features)))
        for j in range(0, len(self.MB_features)):
            for i in range(0, len(self.directories)):
                MB_num = 1
                MB_den = 2 + len(self.files[i])
                for k in self.dictionary.keys():
                    if self.class_train[k - 1] == self.directories[i]:
                        if self.MB_features[j] in self.dictionary[k].keys():
                            # ... e che contengono la parola j
                            MB_num = MB_num + 1
                MB_p_of_w[i, j] = (MB_num / MB_den)
        self.MB_p_of_w = MB_p_of_w

    def calc_p_of_d_given_c(self):
        MB_p_of_d_given_c = []
        for f in range(len(self.dictionary_test)):
            row = np.zeros(len(self.directories), dtype=np.longdouble)
            for w_index in range(len(self.MB_features)):
                if self.MB_features[w_index] in self.dictionary_test[f + 1].keys():
                    row = [row[i] + math.log10(self.MB_p_of_w[i, w_index]) for i in range(len(self.directories))]
                else:
                    row = [row[i] + math.log10(1 - self.MB_p_of_w[i, w_index]) for i in
                           range(len(self.directories))]
            MB_p_of_d_given_c.append(row)
        self.MB_p_of_d_given_c = MB_p_of_d_given_c

    def predict(self):
        MB_TrueCategory = [0 for i in range(len(self.directories))]
        MB_FalseCategory = [0 for i in range(len(self.directories))]
        MB_clf = ['' for i in range(len(self.dictionary_test.keys()))]

        if (self.save):
            self.storage_accuracy = dict()
        else:
            with open('MB_accuracy', 'rb') as f:
                self.storage_accuracy = pickle.load(f)

        for f in range(len(self.dictionary_test)):
            MB_clf_prob = np.zeros(len(self.directories), dtype=np.longdouble)

            for category in range(len(self.directories)):
                MB_clf_prob[category] = (math.log10(self.prior[category]) + self.MB_p_of_d_given_c[f][category])
            # print(MB_clf_prob, '\n')
            MB_estimated_category = np.argmax(MB_clf_prob)

            MB_clf[f] = self.directories[MB_estimated_category]

            if MB_clf[f] == self.class_test[f]:
                MB_TrueCategory[MB_estimated_category] = MB_TrueCategory[MB_estimated_category] + 1
            else:
                MB_FalseCategory[MB_estimated_category] = MB_FalseCategory[MB_estimated_category] + 1
        MB_accuracy = self.get_results(MB_TrueCategory, MB_FalseCategory)
        self.storage_accuracy[self.n] = MB_accuracy
        with open('MB_accuracy', "wb") as f:
            pickle.dump(self.storage_accuracy, f)
        return self.storage_accuracy

    def get_results(self, MB_TrueCategory, MB_FalseCategory):
        t = 0
        f = 0
        print('\nRESULTS WITH N = ', self.n)
        print('\nMulti-variate Bernoulli: ')
        for MB_estimated_category in range(len(self.directories)):
            t = t + MB_TrueCategory[MB_estimated_category]
            f = f + MB_FalseCategory[MB_estimated_category]
            print(self.directories[MB_estimated_category], MB_TrueCategory[MB_estimated_category],
                  MB_FalseCategory[MB_estimated_category])
        MB_accuracy = t / (t + f)
        print('\nAccuracy: ', MB_accuracy)
        return MB_accuracy

