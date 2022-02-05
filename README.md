# Elaborato_AI
Multinomial &amp; Multi-variate Bernoulli Naive Bayes Text Classifier
# Naive Bayes Text Classifier
Questo elaborato implementa due algoritmi per la classificazione del testo: Multinomial & Multi-variate Bernoulli. Il funzionamento di questi algoritmi è descritto da McCallum & Nigam (://www.cs.cmu.edu/~textlearning/)
## Cosa contiene l'elaborato?
L'elaborato contiene 4 file .py: **Multi-variateBernoulliNB.py** conmtiene la classe che implementa l'omonimo algoritmo, allo stesso modo di **MultinomialNB.py**; **aux_function.py** contiene tutte le funzione dedicate alla preparazione del dataset, e l'ultimo file .py è il **main.py**
Alcuni documenti sono troppo pesanti per caricarli quindi la prima volta che si esegue il programma bisogna assegnare save = True, dopo la prima esecuzione lo si può riassegnare a False, si saranno creati tutti i documenti necessari per poter saltare tutti gli step iniziali. Per riprodurre questa parte iniziale dell'algoritmo è necessario importare ntlk (per usufruire del pacchetto 'stop_words') ed eseguire il comando ntlk.download per installare il pacchetto in questione. 
I restanti file servono a velocizzare l'esecuzione del programma e per avere il grafico finale con tutti i valori. Per cambiare taglia del vocabolario modificare la variabile n del main.

