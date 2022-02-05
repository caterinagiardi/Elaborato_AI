import pickle
import math
from decimal import *
from aux_function import *
import sys


class MultinomialNB:

    def __init__(self, save=False, n=100):
        self.save = save
        self.n = n
        self.M_avg_mutual_info = None
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
        self.predictions = None

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
        with open('all_word_list', 'rb') as f:
            all_word_list = pickle.load(f)
        with open('M_mutual_info_FS', 'rb') as f:
            M_avg_mutual_info = pickle.load(f)
        with open('dictionary_test', 'rb') as f:
            dictionary_test = pickle.load(f)

        self.M_avg_mutual_info = M_avg_mutual_info
        self.u_words = u_words
        self.files = files
        self.filepath_list = filepath_list
        self.directories = directories
        self.class_train = class_train
        self.class_test = class_test
        self.dictionary = dictionary
        self.dictionary_test = dictionary_test
        self.all_word_list = all_word_list
        self.classes = classes
        self.doc_train = doc_train


    def get_features(self):
        M_list = []
        tmp2 = np.matrix(self.M_avg_mutual_info)
        for i in range(self.n):
            index = tmp2.argmax()
            tmp2[(index // len(self.u_words)), index - (index // len(self.u_words) * len(self.u_words))] = -1
            M_list.append(index - (index // len(self.u_words) * len(self.u_words)))
        self.M_features = [self.u_words[M_list[i]] for i in range(len(M_list))]


    def calc_prior(self):
        prior = []
        for category_files in self.files:
            prior.append(len(category_files) / len(self.filepath_list))
        self.prior = prior


    def calc_prob_of_word(self):
        M_p_of_w = np.zeros((len(self.directories), len(self.M_features)))
        for j in range(0, len(self.M_features)):
            num_w_in_category = [0 for i in range(len(self.directories))]
            for i in range(0, len(self.directories)):
                M_num = 1
                M_den = len(self.M_features)
                for k in self.dictionary.keys():
                    if self.class_train[k - 1] == self.directories[i]:
                        M_den = M_den + len(self.dictionary[k])
                        if self.M_features[j] in self.dictionary[k].keys():
                            M_num = M_num + self.dictionary[k][self.M_features[j]]
                M_p_of_w[i, j] = Decimal(M_num / M_den)
        self.M_p_of_w = M_p_of_w


    def calc_p_of_d_given_c(self):
        M_p_of_d_given_c = []
        for f in range(len(self.dictionary_test)):
            row2 = [Decimal('0') for s in range(len(self.directories))]
            for w_index in range(len(self.M_features)):
                if self.M_features[w_index] in self.dictionary_test[f + 1].keys():
                    for i in range(len(self.directories)):
                        occ = self.dictionary_test[f + 1][self.M_features[w_index]]
                        if Decimal(self.M_p_of_w[i, w_index]) == Decimal('0'):
                            self.M_p_of_w[i, w_index] = Decimal(sys.float_info.min)
                        if Decimal(pow(self.M_p_of_w[i, w_index], occ)) != Decimal('0'):
                            power = math.log10(Decimal(pow(self.M_p_of_w[i, w_index], occ)))
                        else:
                            power = math.log10(Decimal(sys.float_info.min))
                        row2[i] = Decimal(row2[i]) + Decimal(power)
            M_p_of_d_given_c.append(row2)
        self.M_p_of_d_given_c = M_p_of_d_given_c


    def predict(self):
        M_TrueCategory = [0 for i in range(len(self.directories))]
        M_FalseCategory = [0 for i in range(len(self.directories))]
        M_clf = ['' for i in range(len(self.dictionary_test.keys()))]

        if (self.save):
            self.storage_accuracy = dict()
        else:
            with open('M_accuracy', 'rb') as f:
                self.storage_accuracy = pickle.load(f)
        for f in range(len(self.dictionary_test)):
            M_clf_prob = np.zeros(len(self.directories))
            for category in range(len(self.directories)):
                M_clf_prob[category] = Decimal(math.log(self.prior[category], 10)) + Decimal(
                    self.M_p_of_d_given_c[f][category])
            M_estimated_category = np.argmax(M_clf_prob)

            M_clf[f] = self.directories[M_estimated_category]
            if M_clf[f] == self.class_test[f]:
                M_TrueCategory[M_estimated_category] = M_TrueCategory[M_estimated_category] + 1
            else:
                M_FalseCategory[M_estimated_category] = M_FalseCategory[M_estimated_category] + 1

        M_accuracy = self.get_results(M_TrueCategory, M_FalseCategory)
        self.storage_accuracy[self.n] = M_accuracy
        with open('M_accuracy', "wb") as f:
            pickle.dump(self.storage_accuracy, f)
        return self.storage_accuracy


    def get_results(self, M_TrueCategory, M_FalseCategory):
        t = 0
        f = 0
        print('\nRESULTS WITH N = ', self.n)
        print('\n\nMultinomial: ')
        for M_estimated_category in range(len(self.directories)):
            t = t + M_TrueCategory[M_estimated_category]
            f = f + M_FalseCategory[M_estimated_category]
            print(self.directories[M_estimated_category], M_TrueCategory[M_estimated_category],
                  M_FalseCategory[M_estimated_category])
        M_accuracy = t / (t + f)
        print('\nAccuracy: ', M_accuracy)
        return M_accuracy
