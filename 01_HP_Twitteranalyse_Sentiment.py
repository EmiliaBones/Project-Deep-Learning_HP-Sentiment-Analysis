import pandas as pd
import matplotlib.pyplot as plt

# all sentiments Twitter einlesen

books = []

# einlesen
for i in range(1, 8):
    file_path = f'HP_books/HP{i}.csv'
    df = pd.read_csv(file_path, sep=',', encoding='utf-8')
    books.append(df)

df = pd.read_csv('00_HP_all_books_sentiment_twitter_data.csv')

print(df.head())

print(df.describe())

# plotten (nach kapiteln bzw nach allen kapiteln)

df['block'] = df['sentence_index'] // 100

# Mittelwerte berechnen
grouped = df.groupby(['book', 'block'])['sentiment'].mean().reset_index()

# Bücher sortieren und vorbereiten
unique_books = sorted(grouped['book'].unique())
num_plots = 8  # 4x2 Grid

# Subplots erstellen
fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=True)
axes = axes.flatten()

# Jeden Plot einzeln füllen
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
        # Leeres Feld für Platz 8
        ax.axis('off')

plt.tight_layout()
plt.show()

# 1. Versuch
# zusammenfügen mit df- kapitel, mittelwert plotten


# 2. Versuch
# jeden 2. satz plotten?