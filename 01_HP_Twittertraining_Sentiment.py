import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Eigene Sentimentanalyse durchführen
# ● Twitter-Datensatz (Datenaustausch: 'sentiment (Shrivastava on kaggle, Public Domain).csv') mit Pandas
# laden (Tipp: encoding="utf-8")

df_original = pd.read_csv('00_Twitter_Sentiment_Data.csv',sep=',', encoding='utf-8')
print(df_original.columns)
print(df_original)

# ● Zunächst zum Erstellen des Codes mit weniger Datenpunkten arbeiten. Z.B. Mit Pandas sample(1000)

#df_sample = df_original.sample(100000)
#print(df_sample.head())

# ● Pandas-DataFrame in numpy konvertieren

#df = df_sample.to_numpy()
df = df_original.to_numpy()

# ● Spalte 0 als Label festlegen (durch 4 teilen: float)

y = df[:,0]

#print(np.unique(y)) # Werte: 0 und 4

# ● Spalte 5 als Merkmal festlegen

X = df[:,5]

y[y == 4] = 1
print(f'Labels: \n{y}') 
#print(f'Features: \n{X}')

for i in range(len(X)):
    sentence = " ".join(filter(lambda x: x[0] != '@', X[i].split()))
    sentence = sentence.replace('&ampPOOOOOLL', ' ')  # &
    sentence = sentence.replace('&quotPOOOOOLL', ' ')  # "
    sentence = sentence.replace('&ltPOOOOOLL', ' ')  # <
    sentence = sentence.replace('&gtPOOOOOLL', ' ')  # >
    sentence = sentence.replace('POOOOOLL', ' ')

    X[i] = sentence


# ● Datensatz in Training- und Validierungsdaten aufteilen

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0,test_size=0.2)

print(f'Shapes train: {X_train.shape} und {y_train.shape}')
print(f'Shapes test: {X_test.shape} und {y_test.shape}')
print(type(y_train)) # np array

# ● Anschließend mit dem Textvectorization-Layer in Tokens aufteilen indem er zunächst auf die
# Trainingsdaten adaptiert wird (Tipp: maximale Länge der Texte = 25)

max_length = 25
num_words = 10000 # auf die 10000 häufigsten wörter konzentrieren, seltene Wörter weglassen

tokenizer= keras.layers.TextVectorization(
    max_tokens=num_words,
    standardize="lower_and_strip_punctuation",
    split="whitespace",
    ngrams=None,
    output_mode="int",
    output_sequence_length=max_length,
    pad_to_max_tokens=False,
    vocabulary=None,
    idf_weights=None,
    sparse=False,
    ragged=False,
    encoding="utf-8",
    name=None
)

# Den Tokenizer NUR auf die Trainingsdaten fitten, Unterschied zu anderem Preprocessing, beim MinMax-Scaler ist Reihenfolge egal

tokenizer.adapt(X_train)         # nur auf Trainingsdaten! Würde Tokens erstellen für Daten, die er im Training gar nicht sieht, Wörterbuch bleibt kleiner
tokenizer_model = keras.models.Sequential()    
tokenizer_model.add(tokenizer)                 
X_train_tok = tokenizer_model.predict(X_train, verbose=False) 
X_test_tok = tokenizer_model.predict(X_test, verbose=False) 


print(X_train_tok) 
print(f'Shape: {X_train_tok.shape}') # (800,25) - 800 länge X_train und 25 durch max length
  
vocabulary = tokenizer.get_vocabulary()
vocab_size = len(vocabulary)
 
#print(f'Vocabulary: \n{vocabulary}') 
print(f'Size Vocabulary: {vocab_size}') 

# für die y weil sie nicht im tokenizer waren
y_train = y_train.astype(float)
y_test = y_test.astype(float)

# ● Neuronales Netz mit Embedding-Schicht
# ● Weitere LSTM- oder SimpleRNN-Schichten folgen
# ● 1 Ausgabeneuron, Batch-Size ca. 300 – 1000, Wenige Epochen bevor es ins Overfitting geht. Ca. 20
# Minuten Training (für ganzen Datensatz)
# ● Metric: Accuracy für Trainings- und Testdaten


model = keras.models.Sequential()
model.add(keras.layers.Input((max_length,))) # für model summary - überblick
model.add(keras.layers.Embedding(input_dim=len(tokenizer.get_vocabulary()), output_dim=3, mask_zero=True)) # Matrix mit Zeilenindex/Tokennummer und zufällig initialisierten Werten in den Spalten  # 3 Spalten für positiv/negativ und Negation auf anderer Dimension
#model.add(keras.layers.Flatten()) # bräuchte man, wenn man lstm weglässt und gleich ins dense geht                                                                 
model.add(keras.layers.LSTM(100, return_sequences=True))
model.add(keras.layers.LSTM(100, return_sequences=False))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()


print(f'type X_train: {type(X_train)}')
print(f'type y_train: {type(y_train)}')

model.fit(X_train_tok, y_train, epochs=5, batch_size=100, verbose=True, validation_data=(X_test_tok, y_test))

history = model.history.history

plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Test Accuracy')
plt.xlabel('Epoche')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()


plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Test Loss')
plt.xlabel('Epoche')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

train_score = model.evaluate(X_train_tok, y_train)
test_score = model.evaluate(X_test_tok, y_test)

print('Accuracy Train: ', train_score[1]) # 0: loss, 1: accuracy
print('Accuracy Test: ', test_score[1])


# modell speichern
model.save('twitter_HP.keras')

#%%
model = keras.models.load_model('00_model_twitter_HP.keras')

#%%
# ● Eigenes Beispiel beurteilen

HP1_df = pd.read_csv('HP_books/HP1.csv',sep=',', encoding='utf-8')
HP1_df = HP1_df.to_numpy()
X_HP1 = HP1_df[:,1]

HP1_tok = tokenizer_model.predict(X_HP1)
print(HP1_tok)

HP1_vec = tf.cast(HP1_tok, dtype=tf.int32) 


pred_HP1 = model.predict(HP1_vec)
print(pred_HP1)

# Ergebnis anzeigen
print(f'Prediction: {pred_HP1[:,0].mean()}')

sentence_indices = list(range(1, len(pred_HP1) + 1))

# Plot
plt.figure(figsize=(12, 5))
plt.plot(sentence_indices, pred_HP1, color='mediumblue', linewidth=1)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)

plt.title("Sentimentverlauf über Sätze HP1")
plt.xlabel("Satznummer")
plt.ylabel("Sentimentwert")
plt.grid(True)
plt.tight_layout()
plt.show()

#%%

HP2_df = pd.read_csv('HP_books/HP2.csv',sep=',', encoding='utf-8')
HP2_df = HP2_df.to_numpy()
X_HP2 = HP2_df[:,1]

HP2_tok = tokenizer_model.predict(X_HP2)
print(HP2_tok)

HP2_vec = tf.cast(HP2_tok, dtype=tf.int32) 


pred_HP2 = model.predict(HP2_vec)
print(pred_HP2)

# Ergebnis anzeigen
print(f'Prediction: {pred_HP2[:,0].mean()}')

sentence_indices = list(range(1, len(pred_HP2) + 1))

# Plot
plt.figure(figsize=(12, 5))
plt.plot(sentence_indices, pred_HP2, color='mediumblue', linewidth=1)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)

plt.title("Sentimentverlauf über Sätze HP2")
plt.xlabel("Satznummer")
plt.ylabel("Sentimentwert")
plt.grid(True)
plt.tight_layout()
plt.show()


#%%

HP3_df = pd.read_csv('HP_books/HP3.csv',sep=',', encoding='utf-8')
HP3_df = HP3_df.to_numpy()
X_HP3 = HP3_df[:,1]

HP3_tok = tokenizer_model.predict(X_HP3)
print(HP3_tok)

HP3_vec = tf.cast(HP3_tok, dtype=tf.int32) 


pred_HP3 = model.predict(HP3_vec)
print(pred_HP3)

# Ergebnis anzeigen
print(f'Prediction: {pred_HP3[:,0].mean()}')

sentence_indices = list(range(1, len(pred_HP3) + 1))

# Plot
plt.figure(figsize=(12, 5))
plt.plot(sentence_indices, pred_HP3, color='mediumblue', linewidth=1)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)

plt.title("Sentimentverlauf über Sätze HP3")
plt.xlabel("Satznummer")
plt.ylabel("Sentimentwert")
plt.grid(True)
plt.tight_layout()
plt.show()

#%% 

HP4_df = pd.read_csv('HP_books/HP4.csv',sep=',', encoding='utf-8')
HP4_df = HP4_df.to_numpy()
X_HP4 = HP4_df[:,1]

HP4_tok = tokenizer_model.predict(X_HP4)
print(HP4_tok)

HP4_vec = tf.cast(HP4_tok, dtype=tf.int32) 


pred_HP4 = model.predict(HP4_vec)
print(pred_HP4)

# Ergebnis anzeigen
print(f'Prediction: {pred_HP4[:,0].mean()}')

sentence_indices = list(range(1, len(pred_HP4) + 1))

# Plot
plt.figure(figsize=(12, 5))
plt.plot(sentence_indices, pred_HP4, color='mediumblue', linewidth=1)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)

plt.title("Sentimentverlauf über Sätze HP4")
plt.xlabel("Satznummer")
plt.ylabel("Sentimentwert")
plt.grid(True)
plt.tight_layout()
plt.show()

#%%

HP5_df = pd.read_csv('HP_books/HP5.csv',sep=',', encoding='utf-8')
HP5_df = HP5_df.to_numpy()
X_HP5 = HP5_df[:,1]

HP5_tok = tokenizer_model.predict(X_HP5)
print(HP5_tok)

HP5_vec = tf.cast(HP5_tok, dtype=tf.int32) 


pred_HP5 = model.predict(HP5_vec)
print(pred_HP5)

# Ergebnis anzeigen
print(f'Prediction: {pred_HP5[:,0].mean()}')

sentence_indices = list(range(1, len(pred_HP5) + 1))

# Plot
plt.figure(figsize=(12, 5))
plt.plot(sentence_indices, pred_HP5, color='mediumblue', linewidth=1)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)

plt.title("Sentimentverlauf über Sätze HP5")
plt.xlabel("Satznummer")
plt.ylabel("Sentimentwert")
plt.grid(True)
plt.tight_layout()
plt.show()


#%%

HP6_df = pd.read_csv('HP_books/HP6.csv',sep=',', encoding='utf-8')
HP6_df = HP6_df.to_numpy()
X_HP6 = HP6_df[:,1]

HP6_tok = tokenizer_model.predict(X_HP6)
print(HP6_tok)

HP6_vec = tf.cast(HP6_tok, dtype=tf.int32) 


pred_HP6 = model.predict(HP6_vec)
print(pred_HP6)

# Ergebnis anzeigen
print(f'Prediction: {pred_HP6[:,0].mean()}')

sentence_indices = list(range(1, len(pred_HP6) + 1))

# Plot
plt.figure(figsize=(12, 5))
plt.plot(sentence_indices, pred_HP6, color='mediumblue', linewidth=1)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)

plt.title("Sentimentverlauf über Sätze HP6")
plt.xlabel("Satznummer")
plt.ylabel("Sentimentwert")
plt.grid(True)
plt.tight_layout()
plt.show()


#%%

HP7_df = pd.read_csv('HP_books/HP7.csv',sep=',', encoding='utf-8')
HP7_df = HP7_df.to_numpy()
X_HP7 = HP7_df[:,1]

HP7_tok = tokenizer_model.predict(X_HP7)
print(HP7_tok)

HP7_vec = tf.cast(HP7_tok, dtype=tf.int32) 


pred_HP7 = model.predict(HP7_vec)
print(pred_HP7)

# Ergebnis anzeigen
print(f'Prediction: {pred_HP7[:,0].mean()}')

sentence_indices = list(range(1, len(pred_HP7) + 1))

# Plot
plt.figure(figsize=(12, 5))
plt.plot(sentence_indices, pred_HP7, color='mediumblue', linewidth=1)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)

plt.title("Sentimentverlauf über Sätze HP7")
plt.xlabel("Satznummer")
plt.ylabel("Sentimentwert")
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
# predictions für alle filme -> speichern

# abspeichern
all_predictions = [pred_HP1,pred_HP2,pred_HP3,pred_HP4,pred_HP5,pred_HP6,pred_HP7]

all_data = []

for book_num, predictions in enumerate(all_predictions, start=1):
    for idx, value in enumerate(predictions):
        all_data.append({
            "book": book_num,
            "sentence_index": idx,
            "sentiment": float(value)
        })


df = pd.DataFrame(all_data)
df.to_csv("all_books_sentiment.csv", index=False)

# plotten (nach kapiteln bzw nach allen kapiteln)






















