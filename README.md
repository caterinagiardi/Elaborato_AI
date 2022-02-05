# Elaborato_AI
Multinomial &amp; Multi-variate Bernoulli Naive Bayes Text Classifier
# Naive Bayes Text Classifier
Questo elaborato implementa due algoritmi per la classificazione del testo: Multinomial & Multi-variate Bernoulli. Il funzionamento di questi algoritmi è descritto da McCallum & Nigam (://www.cs.cmu.edu/~textlearning/)
## Cosa contiene l'elaborato?
L'elaborato contiene 4 file .py: **Multi-variateBernoulliNB.py** conmtiene la classe che implementa l'omonimo algoritmo, allo stesso modo di **MultinomialNB.py**; **aux_function.py** contiene tutte le funzione dedicate alla preparazione del dataset, e l'ultimo file .py è il **main.py**
I restanti file servono a velocizzare l'esecuzione del programma: vi è lo split del dataset in train e test, l'analisi dei documenti fino a trasformarli in liste di parole, e il calcolo delle matrici di probabilità per eseguire la feature selection. Se si assegna **True** alla variabile save nel main, il contenuto dei file verrà rieseguito e i file verranno sovrascritti. Per riprodurre questa parte dell'algoritmo è necessario importare ntlk (per usufruire del pacchetto 'stop_words') ed eseguire il comando ntlk.download per installare il pacchetto in questione. Se non si vuole rieseguire la parte già salvata nei file, basta eseguire il programma. Per cambiare taglia del vocabolario modificare la variabile n del main.

