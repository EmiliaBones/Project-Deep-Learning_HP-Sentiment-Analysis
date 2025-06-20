
# https://huggingface.co/bhadresh-savani/bert-base-go-emotion

from transformers import pipeline
from transformers import AutoTokenizer, TFBertForSequenceClassification # TF für TensorFlow
from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import os

#%%

#pipe = pipeline("text-classification", model="bhadresh-savani/bert-base-go-emotion")

# modellauswahl (BERT modell mit GoEmotions-datensatz - 27 emotionen + neutral

tokenizer = AutoTokenizer.from_pretrained("bhadresh-savani/bert-base-go-emotion")
model = TFBertForSequenceClassification.from_pretrained("bhadresh-savani/bert-base-go-emotion",from_pt=True) # ursprünglich pytorch-weights

print(model)
print(tokenizer)

#%%

# Test

EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
    'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
    'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness',
    'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

#text = 'All was well' # {'admiration': 0.17583413422107697, 'approval': 0.1897304505109787, 'neutral': 0.42411568760871887}
#text = 'The scar had not pained Harry for nineteen years' # {'disappointment': 0.15736959874629974, 'realization': 0.15458530187606812, 'neutral': 0.3194434940814972}
#text = 'I am extremely famous' # {'admiration': 0.5537556409835815, 'approval': 0.1591930240392685}
#text = 'He helped Fred and George transform the ghoul' # {'caring': 0.2800358533859253, 'neutral': 0.34050989151000977}
#text = 'Hermione nodded' # {'neutral': 0.5371455550193787}
#text = 'A braver man than Vernon Dursley would have quailed under the furious look Hagrid now gave him when Hagrid spoke , his every syllable trembled with rage' # {'amusement': 0.40175747871398926, 'neutral': 0.35172805190086365}"
#text = 'A horrible , half sucking , half moaning sound came out of the square hole , along with an unpleasant smell like open drains' #{'annoyance': 0.15664584934711456, 'disgust': 0.3638341724872589, 'fear': 0.14907769858837128, 'neutral': 0.1209552139043808}
text = 'In the time it took for their coffees to arrive , Roger Davies and his girlfriend started kissing over their sugar bowl .' #'| Harry wished they wouldn’t he felt that Davies was setting a standard with which Cho would soon expect him to compete .| He felt his face growing hot and tried staring out of the window , but it was so steamed up he could not see the street outside .'

# tokenisierung - enthält input ids, attention_mask, die vom tokenizer erzeugt werden
inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True)

print(inputs)
# {'input_ids': <tf.Tensor: shape=(1, 26), dtype=int32, numpy=
# array([[  101,  1037,  9202,  1010,  2431, 13475,  1010,  2431, 22653,
#          2614,  2234,  2041,  1997,  1996,  2675,  4920,  1010,  2247,
#          2007,  2019, 16010,  5437,  2066,  2330, 18916,   102]])>, 'attention_mask': <tf.Tensor: shape=(1, 26), dtype=int32, numpy=
# array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#         1, 1, 1, 1]])>}


# prediction
logits = model(**inputs).logits # unnormalisierte Rohwerte für jede Klasse
probs = tf.sigmoid(logits)[0].numpy() # normalisieren mit sigmoid, erstes Beispiel (1 Satz)


# schwellenwert festlegen und emotion-dict mit pred-Wert anzeige
threshold = 0.02
emotion_dict = {label: float(prob) for label, prob in zip(EMOTION_LABELS, probs) if prob > threshold}

print(emotion_dict)

#%%
######################################################################################
#%%
# erster ansatz
# HP sentimentanalyse
# satzweise: jeder satz wird für sich klassifiziert

books = []

# einlesen
for i in range(1, 8):
    file_path = f'HP_books/HP{i}.csv'
    df = pd.read_csv(file_path, sep=',', encoding='utf-8')
    books.append(df)

print(books[0].head())  # HP1
print(books[0].columns) # chapter, sentence
#print(books[1].head())  # HP2

for i, df in enumerate(books, start=1):
    print(f'Buch {i}: {len(df)} Zeilen')

#%%

EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
    'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
    'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness',
    'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

threshold = 0 # alle emotionslabels aufführen
all_results = []

# tokenizing und predictions für HP1-HP7

for i in range(1, 8):
    file_path = f"HP_books/HP{i}.csv"
    
    print(f'processing {file_path}')
    
    df = pd.read_csv(file_path, sep=",", encoding="utf-8")
    
    texts = df["sentence"].dropna().tolist()
    chapters = df["chapter"].dropna().tolist() 
    
    
    batch_size = 32 # speicher schonen
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        batch_chapters = chapters[start:start + batch_size]
        
        # tokenizer
        inputs = tokenizer(batch_texts, return_tensors="tf", padding=True, truncation=True)
        
        # prediction
        logits = model(**inputs).logits
        probs = tf.sigmoid(logits).numpy()
        
        # df alles zusammenfügen
        for text, chapter, prob in zip(batch_texts, batch_chapters, probs):
            emotions = {label: float(p) for label, p in zip(EMOTION_LABELS, prob) if p > threshold}
            all_results.append({
                'book': f'HP{i}', 
                'chapter': chapter,
                'text': text,
                'emotions': emotions
            })

# ergebnisse als dataframe speichern
results_df = pd.DataFrame(all_results)
results_df.to_csv('HP_BERT+GoEmo.csv', index=False)

#%%
######################################################################################
#%%
# zweiter ansatz
# HP sentimentanalyse mit mehr kontext
# 10-satz batches in eine row

books = []

# einlesen
for i in range(1, 8):
    file_path = f'HP_books/HP{i}.csv'
    df = pd.read_csv(file_path, sep=',', encoding='utf-8')
    books.append(df)


for i, df in enumerate(books, start=1):
    print(f'Buch {i}: {len(df)} Zeilen')


books_grouped = []

# datensatz in mit 10 sätzen als stapel
# für jedes buch werden 10 rows zusammengefasst
# bei 5 übrigen sätzen am ende noch auf vorhergehenden satzstapel, ab 6 sätzen neuen stapel bilden

for book in books:
    # keep order of chapters
    chapter_order = book['chapter'].drop_duplicates().tolist()
    book['chapter'] = pd.Categorical(book['chapter'], categories=chapter_order, ordered=True)

    grouped_rows = []

    for chapter, group in book.groupby('chapter', sort=False):
        sentences = group['sentence'].tolist()
        n = len(sentences)
        i = 0

        while i + 10 <= n:
            chunk = sentences[i:i+10]
            grouped_rows.append({
                'chapter': chapter,
                'sentence': ' '.join(chunk)
            })
            i += 10

        remainder = n - i
        if 0 < remainder <= 5 and grouped_rows:
            grouped_rows[-1]['sentence'] += ' ' + ' '.join(sentences[i:])
        elif remainder >= 6:
            chunk = sentences[i:]
            grouped_rows.append({
                'chapter': chapter,
                'sentence': ' '.join(chunk)
            })

    grouped_df = pd.DataFrame(grouped_rows)
    books_grouped.append(grouped_df)

print(len(books_grouped[1]))

filenames = [f"HP_books/HP{i+1}_grouped.csv" for i in range(len(books_grouped))]

# bücher separat abspeichern
for df, filename in zip(books_grouped, filenames):
    df.to_csv(filename, index=False, encoding='utf-8')

#%%

EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
    'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
    'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness',
    'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

threshold = 0 # alle emotionslabels aufführen
all_results = []

# für alle bücher satzstapel tokenisieren und predicten

for i in range(7,8):
    file_path = f"HP_books/HP{i}_grouped.csv"
    
    print(f'processing {file_path}')
    
    df = pd.read_csv(file_path, sep=",", encoding="utf-8")
    
    texts = df["sentence"].dropna().tolist()
    chapters = df["chapter"].dropna().tolist() 
    
    
    batch_size = 32 # speicher schonen
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        batch_chapters = chapters[start:start + batch_size]
        
        # tokenizer
        inputs = tokenizer(batch_texts, return_tensors="tf", padding=True, truncation=True)
        
        # prediction
        logits = model(**inputs).logits
        probs = tf.sigmoid(logits).numpy()
        
        # df alles zusammenfügen
        for text, chapter, prob in zip(batch_texts, batch_chapters, probs):
            emotions = {label: float(p) for label, p in zip(EMOTION_LABELS, prob) if p > threshold}
            all_results.append({
                'book': f'HP{i}', 
                'chapter': chapter,
                'text': text,
                'emotions': emotions
            })


results_df = pd.DataFrame(all_results)
results_df.to_csv('HP_BERT+GoEmo_10.csv', index=False)
