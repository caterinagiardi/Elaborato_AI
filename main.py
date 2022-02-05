import time
from MultinomialNB import *
from MultivariateBernoulliNB import *


want_MB = True
want_M = True
save = False
feature_probs = False
n = 20
if save:
    prepare_data(feature_probs)
if want_M:
    print('Start Multinomial:')
    M = MultinomialNB(n=n)
    M_accuracy = M.predict()


if want_MB:
    print('\nStart Multivariate.')
    MB = MultivariateBernoulliNB(n=n)
    MB_accuracy = MB.predict()

with open('MB_accuracy', 'rb') as f:
    MB_accuracy = pickle.load(f)
with open('M_accuracy', 'rb') as f:
    M_accuracy = pickle.load(f)

plot_accuracy(M_accuracy, MB_accuracy)

