import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#%%
# Sentimentanalyse
# einfaches LSTM-Modell: Training mit Twitterdatensatz

#%%
# preparation

df_original = pd.read_csv('00_Twitter_Sentiment_Data.csv',sep=',', encoding='utf-8')
print(df_original.columns)
print(df_original)

# für weniger datenpunkte

# df_sample = df_original.sample(100000)
# print(df_sample.head())

# konvertieren
#df = df_sample.to_numpy()
df = df_original.to_numpy()

# labels und Daten
X = df[:,5]
y = df[:,0]

# 4 durch 1 ersetzen
y[y == 4] = 1

print(f'Labels: \n{y}') 
#print(f'Features: \n{X}')

#%%

# sonderzeichen herausfiltern
for i in range(len(X)):
    sentence = " ".join(filter(lambda x: x[0] != '@', X[i].split()))
    sentence = sentence.replace('&ampPOOOOOLL', ' ')  # &
    sentence = sentence.replace('&quotPOOOOOLL', ' ')  # "
    sentence = sentence.replace('&ltPOOOOOLL', ' ')  # <
    sentence = sentence.replace('&gtPOOOOOLL', ' ')  # >
    sentence = sentence.replace('POOOOOLL', ' ')

    X[i] = sentence

#%%
# Datensatz in Training- und Validierungsdaten aufteilen

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0,test_size=0.2)

print(f'Shapes train: {X_train.shape} und {y_train.shape}')
print(f'Shapes test: {X_test.shape} und {y_test.shape}')
print(type(y_train)) # np array

#%%
# trainingsdaten mit textvectorization-layer in tokens aufteilen

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

# tokenizer NUR auf die trainingsdaten fitten (unterschied zu anderem preprocessing)

tokenizer.adapt(X_train)         # nur auf trainingsdaten! würde Tokens erstellen für daten, die er im training gar nicht sieht

tokenizer_model = keras.models.Sequential()    
tokenizer_model.add(tokenizer)

X_train_tok = tokenizer_model.predict(X_train, verbose=False) 
X_test_tok = tokenizer_model.predict(X_test, verbose=False) 

#%%
print(X_train_tok) 
print(f'Shape: {X_train_tok.shape}')
  
vocabulary = tokenizer.get_vocabulary()
vocab_size = len(vocabulary)
 
#print(f'Vocabulary: \n{vocabulary}') 
print(f'Size Vocabulary: {vocab_size}') 

#%%
# konvertieren weil die y sie nicht im tokenizer waren
y_train = y_train.astype(float)
y_test = y_test.astype(float)


#%%
# NN mit embedding layer - output_dim=3 für positiv, negativ und negation ausreichend in dem Fall
# 2 LSTM-schichten
# 1 ausgabeneuron für positiv/negativ
# metric accuracy, binary crossentropy für binäre klassifikation

model = keras.models.Sequential()
model.add(keras.layers.Input((max_length,))) # für model summary
model.add(keras.layers.Embedding(input_dim=len(tokenizer.get_vocabulary()), output_dim=3, mask_zero=True)) # zufällige initialisierung
model.add(keras.layers.LSTM(100, return_sequences=True))
model.add(keras.layers.LSTM(100, return_sequences=False))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# kleine batchsize, wenige epochen, sonst overfitting
# fit mit X_train und validation mit X_test
model.fit(X_train_tok, y_train, epochs=5, batch_size=100, verbose=True, validation_data=(X_test_tok, y_test))

#%%
# accuracy und loss plotten

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

#%%
## scores speichern

train_score = model.evaluate(X_train_tok, y_train)
test_score = model.evaluate(X_test_tok, y_test)

print('Accuracy Train: ', train_score[1]) # 0: loss, 1: accuracy
print('Accuracy Test: ', test_score[1])

#%%
# modell speichern
model.save('twitter_HP.keras')

# modell laden
model = keras.models.load_model('00_model_twitter_HP.keras')

#%%
# anwenden auf HP-Datensätze

# HP1
HP1_df = pd.read_csv('HP_books/HP1.csv',sep=',', encoding='utf-8')
HP1_df = HP1_df.to_numpy()
X_HP1 = HP1_df[:,1]

HP1_tok = tokenizer_model.predict(X_HP1)
print(HP1_tok)

HP1_vec = tf.cast(HP1_tok, dtype=tf.int32) 

# results
pred_HP1 = model.predict(HP1_vec)
print(pred_HP1)

print(f'Prediction: {pred_HP1[:,0].mean()}')

sentence_indices = list(range(1, len(pred_HP1) + 1))

# plot
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
# HP2

HP2_df = pd.read_csv('HP_books/HP2.csv',sep=',', encoding='utf-8')
HP2_df = HP2_df.to_numpy()
X_HP2 = HP2_df[:,1]

HP2_tok = tokenizer_model.predict(X_HP2)
print(HP2_tok)

HP2_vec = tf.cast(HP2_tok, dtype=tf.int32) 


pred_HP2 = model.predict(HP2_vec)
print(pred_HP2)

# results
print(f'Prediction: {pred_HP2[:,0].mean()}')

sentence_indices = list(range(1, len(pred_HP2) + 1))

# plot
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
# HP3

HP3_df = pd.read_csv('HP_books/HP3.csv',sep=',', encoding='utf-8')
HP3_df = HP3_df.to_numpy()
X_HP3 = HP3_df[:,1]

HP3_tok = tokenizer_model.predict(X_HP3)
print(HP3_tok)

HP3_vec = tf.cast(HP3_tok, dtype=tf.int32) 


pred_HP3 = model.predict(HP3_vec)
print(pred_HP3)

# results
print(f'Prediction: {pred_HP3[:,0].mean()}')

sentence_indices = list(range(1, len(pred_HP3) + 1))

# plot
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
# HP4

HP4_df = pd.read_csv('HP_books/HP4.csv',sep=',', encoding='utf-8')
HP4_df = HP4_df.to_numpy()
X_HP4 = HP4_df[:,1]

HP4_tok = tokenizer_model.predict(X_HP4)
print(HP4_tok)

HP4_vec = tf.cast(HP4_tok, dtype=tf.int32) 


pred_HP4 = model.predict(HP4_vec)
print(pred_HP4)

# results
print(f'Prediction: {pred_HP4[:,0].mean()}')

sentence_indices = list(range(1, len(pred_HP4) + 1))

# plot
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
# HP5

HP5_df = pd.read_csv('HP_books/HP5.csv',sep=',', encoding='utf-8')
HP5_df = HP5_df.to_numpy()
X_HP5 = HP5_df[:,1]

HP5_tok = tokenizer_model.predict(X_HP5)
print(HP5_tok)

HP5_vec = tf.cast(HP5_tok, dtype=tf.int32) 


pred_HP5 = model.predict(HP5_vec)
print(pred_HP5)

# results
print(f'Prediction: {pred_HP5[:,0].mean()}')

sentence_indices = list(range(1, len(pred_HP5) + 1))

# plot
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
# HP6

HP6_df = pd.read_csv('HP_books/HP6.csv',sep=',', encoding='utf-8')
HP6_df = HP6_df.to_numpy()
X_HP6 = HP6_df[:,1]

HP6_tok = tokenizer_model.predict(X_HP6)
print(HP6_tok)

HP6_vec = tf.cast(HP6_tok, dtype=tf.int32) 


pred_HP6 = model.predict(HP6_vec)
print(pred_HP6)

# results
print(f'Prediction: {pred_HP6[:,0].mean()}')

sentence_indices = list(range(1, len(pred_HP6) + 1))

# plot
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
# HP7

HP7_df = pd.read_csv('HP_books/HP7.csv',sep=',', encoding='utf-8')
HP7_df = HP7_df.to_numpy()
X_HP7 = HP7_df[:,1]

HP7_tok = tokenizer_model.predict(X_HP7)
print(HP7_tok)

HP7_vec = tf.cast(HP7_tok, dtype=tf.int32) 


pred_HP7 = model.predict(HP7_vec)
print(pred_HP7)

# results
print(f'Prediction: {pred_HP7[:,0].mean()}')

sentence_indices = list(range(1, len(pred_HP7) + 1))

# plot
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
# predictions für alle bücher speichern

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

#%%

# HP results all sentiments Twitter einlesen

books = []

# einlesen
for i in range(1, 8):
    file_path = f'HP_books/HP{i}.csv'
    df = pd.read_csv(file_path, sep=',', encoding='utf-8')
    books.append(df)

df = pd.read_csv('00_HP_all_books_sentiment_twitter_data.csv')

print(df.head())
print(df.describe())

# %%

# plotten mittelwert nach 100 sätzen

df['block'] = df['sentence_index'] // 100

grouped = df.groupby(['book', 'block'])['sentiment'].mean().reset_index()

unique_books = sorted(grouped['book'].unique())
num_plots = 8  # 4x2 Grid

# subplots
fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=True)
axes = axes.flatten()

for i in range(num_plots):
    ax = axes[i]

    if i < len(unique_books):
        book_id = unique_books[i]
        book_data = grouped[grouped['book'] == book_id]

        ax.plot(book_data['block'], book_data['sentiment'], color='blue')
        ax.set_title(f"HP{book_id}")
        ax.set_ylim(0.15, 0.6)
        ax.set_xlabel("Textabschnitt (100 Sätze)")
        if i % 4 == 0:
            ax.set_ylabel("Sentiment mean")
        ax.grid(True)
    else:
        ax.axis('off')

plt.tight_layout()
plt.show()




















