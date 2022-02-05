import time
from MultinomialNB import *
from MultivariateBernoulliNB import *


want_MB = False
want_M = False
save = False
n = 20
if save:
    prepare_data(save)
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
print(M_accuracy)
print(MB_accuracy)