
import string
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from nltk.corpus import *
import os
import pickle
import time
import math
from sklearn.model_selection import train_test_split

# FUNCTION DEFINITION SPACE
def clean_words(words):
    table = str.maketrans('', '', '\t')
    words = [word.translate(table) for word in words]
    punctuations = (string.punctuation)
    trans_table = str.maketrans('', '', punctuations)
    stripped_words = [word.translate(trans_table) for word in words]
    words = [str for str in stripped_words if str]
    for word in words:
        word.strip("'")
        word.strip('"')
    words = [word for word in words if not word.isdigit()]
    words = [word for word in words if not len(word) == 1]
    words = [str for str in words if str]
    words = [word.lower() for word in words]
    words = [word for word in words if len(word) > 3]
    return words

def remove_stopwords(words):
    stop_words = set(stopwords.words('english'))
    r = []
    for word in words:
        if word not in stop_words:
            r.append(word)
    return r

def line_tokenizer(line):
    # per ogni riga del documento individuo le parole e le pulisco, dopodiché selezione le parole perché siano consistenti
    words = line[0:len(line) - 1].strip().split(" ")
    words = clean_words(words)
    words = remove_stopwords(words)
    return words

def remove_metadata(lines):
    for i in range(len(lines)):
        if (lines[i] == '\n'):
            start = i + 1
            break
    new_lines = lines[start:]
    return new_lines

def tokenizer(doc_path):
    # apro il documento in lettura e le righe del file le salvo in lines
    file = open(doc_path, 'r')
    lines = file.readlines()
    lines = remove_metadata(lines)

    doc_words = []
    # con l'aiuto della fuction line_tokenizer() individuo le parole di ogni riga le inserisco in una lista (avrò una
    # lista per ogni riga) che andrò ad inserire in doc_words
    for line in lines:
        doc_words.append(line_tokenizer(line))
    return doc_words

def to_1D(list):
    new_list = []
    # list è bidimensionale e quindi l'iteratore i è una lista a sua volta; salvo ogni elemento j della lista i in un nuovo array
    for i in list:
        for j in i:
            new_list.append(j)
    return new_list

def plot_accuracy(M_accuracy, MB_accuracy):
    x1 = np.asarray(sorted(M_accuracy.keys()))
    y1 = np.zeros(len(x1))

    x2 = np.asarray(sorted(MB_accuracy.keys()))
    y2 = np.zeros(len(x2))
    for i in range(len(x2)):
        y1[i] = M_accuracy[x1[i]]
        y2[i] = MB_accuracy[x2[i]]

    blue = mlines.Line2D([], [], color='blue', label = 'Multinomial')
    red = mlines.Line2D([], [], color='red', label = 'Multi-variate Bernouolli')
    plt.legend(handles = [blue, red], title = 'Legenda:')
    plt.xscale('log')
    plt.plot(x1, y1, 'b', x2, y2, 'r')
    plt.title("Plot")
    plt.xlabel("n")
    plt.ylabel("accuracy")
    plt.savefig("Plot generated using Matplotlib.png")
    plt.show()

    print(x2, ' \n',  y2)

    print('\n \n', x1, '\n', y1)

def prepare_data():
    root = '20_newsgroups'
    directories = [dir for dir in os.listdir(root)]
    files = []
    for dir_name in directories:
        dir = os.path.join(root, dir_name)
        files.append([file for file in os.listdir(dir)])
    # files è un array bidimensionale dove ad ogni riga corrisponde una lista di files. In sostanza ho una suddivisione dei
    # file per cartella
    filepath_list = []
    # filepath_list contiene l'insieme di path di tutti i documenti
    for dir_index in range(len(directories)):
        for file in files[dir_index]:
            # creo il path e lo aggiungo alla lista
            filepath_list.append(os.path.join(root, os.path.join(directories[dir_index], file)))

    # lista che contiene la classe di appartenenza di ogni documento
    classes = []
    for dir_name in directories:
        folder_path = os.path.join(root, dir_name)
        num_of_files = len(os.listdir(folder_path))
        for i in range(num_of_files):
            classes.append(dir_name)

    print('Divido il dataset.')
    doc_train, doc_test, class_train, class_test = train_test_split(filepath_list, classes, random_state=0,
                                                                    test_size=0.20)

    # per ogni doc del train set aggiungo alla lista le parole che compaiono in quel documento
    word_list = []
    for document in doc_train:
        word_list.append(to_1D(tokenizer(document)))

    with open('word_list', "wb") as f:
        pickle.dump(word_list, f)
    with open('files', "wb") as f:
        pickle.dump(files, f)
    with open('directories', "wb") as f:
        pickle.dump(directories, f)
    with open('filepath_list', 'wb') as f:
        pickle.dump(filepath_list, f)
    with open('classes', 'wb') as f:
        pickle.dump(classes, f)
    with open('doc_train', "wb") as f:
        pickle.dump(doc_train, f)
    with open('class_train', "wb") as f:
        pickle.dump(class_train, f)
    with open('doc_test', "wb") as f:
        pickle.dump(doc_test, f)
    with open('class_test', "wb") as f:
        pickle.dump(class_test, f)



    # a questo punto word_list contiene per ogni riga tutte le parole di ogni documento del train set
    print(len(word_list))

    all_word_list = np.asarray(to_1D(word_list))
    print(len(all_word_list))

    # adesso troviamo le unique words dei documenti del train set
    u_words, counts = np.unique(all_word_list, return_counts=True)
    print('unique: ', len(u_words))

    tmp = u_words.copy()
    u_words = []
    # rimuovo le parole che compaiono solo una volta nel vocabolario
    for i in range(len(tmp)):
        if counts[i] > 1:
            u_words.append(tmp[i])
    print(len(u_words))

    # creo un dizionario di dizionari che per ogni documento (identificato con un numero pari all'ordine di lettura) ha un
    # dizionario con tutte le parole. In sostanza: creo un vocabolario con le unique word di ogni documento e lo riempio con
    # le volte che quella parola compare in quel documento.
    dictionary = {}
    doc_num = 1
    for doc_words in word_list:
        np_doc_words = np.asarray(doc_words)
        w, c = np.unique(np_doc_words, return_counts=True)
        dictionary[doc_num] = {}
        for i in range(len(w)):
            dictionary[doc_num][w[i]] = c[i]
        doc_num = doc_num + 1

    with open('unique_words', 'wb') as f:
        pickle.dump(u_words, f)
    with open('dictionary', 'wb') as f:
        pickle.dump(dictionary, f)
    with open('all_word_list', 'wb') as f:
        pickle.dump(all_word_list, f)

    M_category_probs = []
    M_word_probs = []
    MB_category_probs = []
    MB_word_probs = []

    for category_files in files:
        MB_category_probs.append(len(category_files) / len(filepath_list))

    for category_index in range(len(directories)):
        count = 0
        for doc_index in range(len(word_list)):
            if class_train[doc_index] == directories[category_index]:
                count = count + len(word_list[doc_index])
        M_category_probs.append(count / len(all_word_list))

    M_p_word_and_category = []
    MB_p_word_and_category = []
    n = 0
    for p_word in u_words:
        MB_count = [0, [0 for i in range(len(directories))]]
        M_count = [0, [0 for i in range(len(directories))]]

        for i in dictionary.keys():
            if p_word in dictionary[i].keys():
                MB_count[0] = MB_count[0] + 1
                M_count[0] = M_count[0] + dictionary[i][p_word]
                for category in range(len(directories)):
                    if class_train[i - 1] == directories[category]:
                        MB_count[1][category] = MB_count[1][category] + 1
                        M_count[1][category] = M_count[1][category] + dictionary[i][p_word]
        tmp = MB_count[1].copy()
        MB_count[1] = [(tmp[k] / len(filepath_list)) for k in range(len(tmp))]
        tmp = M_count[1].copy()
        M_count[1] = [(tmp[k] / len(filepath_list)) for k in range(len(tmp))]

        MB_p_word_and_category.append(MB_count[1])
        M_p_word_and_category.append(M_count[1])

        MB_word_probs.append(MB_count[0] / len(filepath_list))
        M_word_probs.append(M_count[0] / len(all_word_list))
        n = n + 1

    MB_avg_mutual_info = []
    M_avg_mutual_info = []
    M_word_probs = [M_word_probs[i] * 1000000 for i in range(len(M_word_probs))]
    M_word_probs = np.array(M_word_probs, dtype=np.longdouble)
    MB_word_probs = np.array(MB_word_probs, dtype=np.longdouble)
    M_p_word_and_category = np.matrix(M_p_word_and_category, dtype=np.longdouble)
    print('M_p_word and category: ', np.shape(M_p_word_and_category))
    for j in range(len(directories)):
        M_sum = [0 for i in range(len(u_words))]
        MB_sum = [0 for i in range(len(u_words))]
        for i in range(len(u_words)):
            if (M_p_word_and_category[i, j] != 0):
                M_log = (M_p_word_and_category[i, j]) / (M_word_probs[i] * M_category_probs[j])
                M_log = M_log * 1000000
                M_sum[i] = M_p_word_and_category[i, j] * math.log(M_log, 10)
            else:
                M_sum[i] = 0
            if (MB_p_word_and_category[i][j] != 0):
                den = MB_category_probs[j] * MB_word_probs[i] * 10000
                MB_log = (MB_p_word_and_category[i][j] * 10000) / (den)
                MB_sum[i] = MB_p_word_and_category[i][j] * math.log(MB_log, 10)
            else:
                MB_sum[i] = 0
        M_avg_mutual_info.append(M_sum)
        MB_avg_mutual_info.append(MB_sum)

    print('Fine calcolo probabilità per features.')
    with open('M_mutual_info_FS', 'wb') as f:
        pickle.dump(M_avg_mutual_info, f)
    with open('MB_mutual_info_FS', 'wb') as f:
        pickle.dump(MB_avg_mutual_info, f)

    # inizio ad elaborare il set dei test
    list_of_words_test = []
    for document in doc_test:
        list_of_words_test.append(to_1D(tokenizer(document)))

    dictionary_test = {}
    doc_num = 1
    for doc_words in list_of_words_test:
        np_doc_words = np.asarray(doc_words)
        w, c = np.unique(np_doc_words, return_counts=True)
        dictionary_test[doc_num] = {}
        for i in range(len(w)):
            dictionary_test[doc_num][w[i]] = c[i]
        doc_num = doc_num + 1


    with open('dictionary_test', 'wb') as f:
        pickle.dump(dictionary_test, f)
    print('Fine preparazione dati')