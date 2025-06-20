import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast

import os
print("Aktuelles Arbeitsverzeichnis:", os.getcwd())

books = []

# einlesen
for i in range(1, 8):
    file_path = f'HP{i}_BERT+GoEmo_10.csv'
    df = pd.read_csv(file_path, sep=',', encoding='utf-8')
    df['book'] = f'HP{i}'
    books.append(df)

print(type(books))
print(books[0].head())

HP1_df = books[0]
HP2_df = books[1]
HP3_df = books[2]
HP4_df = books[3]
HP5_df = books[4]
HP6_df = books[5]
HP7_df = books[6]


#%%

HP1_chapters = ['The Boy Who Lived', 'The Vanashing Glass','The Letters From No One','The Keeper of the Keys','Diagon Alley','The Journey from Platform Nine and Three Quarters','The Sorting Hat','The Potions Master','The Midnight Duel','Halloween','Quidditch','The Mirror of Erised','Nicolas Flamel','Norbert the Norwegian Ridgeback','The Forbidden Forest','Through the Trapdoor','The Man with Two Faces']
HP2_chapters = ['The Worst Birthday','Dobby’s Warning','The Burrow','At Flourish and Blotts','The Whomping Willow','Gilderoy Lockhart','Mudbloods and Murmurs','The Deathday Party','The Writing on the Wall','The Rogue Bludger','The Dueling Club','The Poly Juice Potion','The Very Secret Diary','Cornelius Fudge','Aragog','The Chamber of Secrets','The Heir of Slytherin','Dobby’s Reward']
HP3_chapters = ['Owl Post','Aunt Marge’s Big Mistake','The Knight Bus','The Leaky Cauldron','The Dementor','Talons and Tea Leaves','The Boggart in the Wardrobe','Flight of the Fat Lady','Grim Defeat','The Marauder’s Map','The Firebolt','The Patronus','Gryffindor Versus Ravenclaw','Snape’s Grudge','The Quidditch Final','Professor Trelawney’s Prediction','Cat, Rat and Dog','Moony, Wormtail, Padfoot, and Prongs','The Servant of Lord Voldemort','The Dementor’s Kiss','Hermione’s Secret','Owl Post Again']
HP4_chapters = ['The Riddle House','The Scar','The Invitation','Back to the Burrow','Weasley’s Wizard Wheezes','The Portkey','Bagman and Crouch','The Quidditch World Cup','The Dark Mark','Mayhem at the Ministry','Aboard the Hogwarts Express','The Triwizard Tournament','Mad-Eye Moody','The Unforgivable Curses','Beauxbatons and Durmstrang','The Goblet of Fire','The Four Champions','The Weighing of the Wands','The Hungarian Horntail','The First Task','The House-Elf Liberation Front','The Unexpected Task','The Yule Ball','Rita Skeeter’s Scoop','The Egg and the Eye','The Second Task','Padfoot Returns','The Madness of Mr Crouch','The Dream','The Pensieve','The Third Task','Flesh, Blood, and Bone','The Death Eaters','Priori Incantatem','Veritaserum','The Parting of the Ways','The Beginning']
HP5_chapters = ['Dudley Demented','A Peck of Owls','The Advanced Guard','Number Twelve, Grimmauld Place','The Order of the Phoenix','The Noble and Most Ancient House of Black','The Ministry of Magic','The Hearing','The Woes of Mrs Weasley','Luna Lovegood','The Sorting Hat’s New Song','Professor Umbridge','Detention with Dolores','Percy and Padfoot','The Hogwarts High Inquisitor','In The Hog’s Head','Educational Decree Number Twenty-Four','Dumbledore’s Army','The Lion and the Serpent','Hagrid’s Tale','The Eye of the Snake','St. Mungo’s Hospital for Magical Maladies and Injuries','Christmas on the Closed Ward','Occlumency','The Beetle at Bay','Seen and Unforeseen','The Centaur and the Sneak','Snape’s Worst Memory','Careers Advice','Grawp','O.W.L.s','Out of the Fire']
HP6_chapters = ['The Other Minister','Spinner’s End','Will and Won’t','Horace Slughorn','An Excess of Phlegm','Draco’s Detour','The Slug Club','Snape Victorious','The Half Blood Prince','The House of Gaunt','Hermione’s Helping Hand','Silver and Opals','The Secret Riddle','Felix Felicis','The Unbreakable Vow','A Very Frosty Christmas','A Sluggish Memory','Birthday Surprises','Elf Tails','Lord Voldemort’s Request','The Unknowable Room','After the Burial','Horcruxes','Sectumsempra','The Seer Overheard','The Cave','The Lightning Struck Tower','Flight of the Prince','The Phoenix Lament','The White Tomb']
HP7_chapters = ['The Dark Lord Ascending','In Memoriam','The Dursleys Departing','The Seven Potters','Fallen Warrior','The Ghoul in Pajamas','The Will of Albus Dumbledore','The Wedding','A Place to Hide','Kreacher’s Tale','The Bribe','Magic is Might','The Muggle Born Registration Commission','The Thief','The Goblin’s Revenge','Godric’s Hollow','Bathilda’s Secret','The Life and Lies of Albus Dumbledore','The Silver Doe','Xenophilius Lovegood','The Tale of the Three Brothers','The Deathly Hallows','Malfoy Manor','The Wandmaker','Shell Cottage','Gringotts','The Final Hiding Place','The Missing Mirror','The Lost Diadem','The Sacking of Severus Snape','The Battle of Hogwarts','The Elder Wand','The Prince’s Tale','The Forest Again','King’s Cross','The Flaw in the Plan','Nineteen Years Later']

# alle kapitelüberschriften in uppercase
HP1_chapters = [chapter.upper() for chapter in HP1_chapters]
HP2_chapters = [chapter.upper() for chapter in HP2_chapters]
HP3_chapters = [chapter.upper() for chapter in HP3_chapters]
HP4_chapters = [chapter.upper() for chapter in HP4_chapters]
HP5_chapters = [chapter.upper() for chapter in HP5_chapters]
HP6_chapters = [chapter.upper() for chapter in HP6_chapters]
HP7_chapters = [chapter.upper() for chapter in HP7_chapters]

EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
    'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
    'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness',
    'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

#%%

for book in books:
    book['emotions'] = book['emotions'].apply(ast.literal_eval)
    
#%%


HP_dfs = [HP1_df, HP2_df, HP3_df, HP4_df, HP5_df, HP6_df, HP7_df]
HP_chapters_list = [HP1_chapters, HP2_chapters, HP3_chapters, HP4_chapters, HP5_chapters, HP6_chapters, HP7_chapters]
EMOTION_LABELS = ['joy', 'fear', 'sadness', 'anger', 'disgust', 'surprise', 'neutral'] # vorauswahl von mögl. relevanten emotionen

fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharey=True)
axes = axes.flatten()

for i in range(7):
    df = HP_dfs[i]
    chapters = HP_chapters_list[i]
    
    if not all(label in df.columns for label in EMOTION_LABELS):
        df_emo = df['emotions'].apply(pd.Series)
        df = pd.concat([df, df_emo], axis=1)
    
    df_chap = df.groupby('chapter')[EMOTION_LABELS].mean()
    df_chap = df_chap.loc[chapters]
    
    ax = axes[i]
    
    chapter_nums = list(range(1, len(df_chap) + 1))
    
    for emotion, color in zip(['joy', 'fear', 'sadness', 'anger'], ['green', 'red', 'blue', 'yellow']):
        ax.plot(chapter_nums, df_chap[emotion], label=emotion.capitalize(), color=color, alpha=0.6)
    
    ax.set_xticks(chapter_nums)
    ax.set_xticklabels(chapter_nums, rotation=0, fontsize=8)
    ax.set_ylim(0, 0.05)
    ax.set_title(f'Sentiment by chapter – HP{i+1}')
    ax.grid(True)

axes[7].axis('off')

fig.text(0.06, 0.5, 'Prediction', va='center', rotation='vertical', fontsize=12)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

#%%
# emotionen clustern

positive = ['admiration', 'amusement','approval', 'caring', 'desire', 'excitement', 'gratitude', 'joy', 'love',  'optimism', 'pride', 'relief']
negative = ['anger', 'annoyance', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'fear',  'grief', 'nervousness', 'remorse', 'sadness']
ambiguous = ['confusion', 'curiosity', 'realization','surprise']

chapters_all = [HP1_chapters, HP2_chapters, HP3_chapters, HP4_chapters, HP5_chapters, HP6_chapters, HP7_chapters]

def group_emotions(emotions_dict):
    return {
        'positive': sum(emotions_dict.get(em, 0) for em in positive),
        'negative': sum(emotions_dict.get(em, 0) for em in negative),
        'ambiguous': sum(emotions_dict.get(em, 0) for em in ambiguous),
        'neutral': emotions_dict.get('neutral', 0)
    }

dfs = [HP1_df, HP2_df, HP3_df, HP4_df, HP5_df, HP6_df, HP7_df]

for i in range(len(dfs)):
    grouped_emo = dfs[i]['emotions'].apply(group_emotions).apply(pd.Series)
    dfs[i] = pd.concat([dfs[i].drop(columns=['emotions']), grouped_emo], axis=1)

titles = ['HP1', 'HP2', 'HP3', 'HP4', 'HP5', 'HP6', 'HP7']
EMOTION_LABELS_4 = ['positive', 'negative']
colors = ['green', 'red']

fig, axes = plt.subplots(2, 4, figsize=(24, 10), sharey=True)
axes = axes.flatten()

for i, (df, chapters_order) in enumerate(zip(dfs, chapters_all)):
    df['chapter'] = pd.Categorical(df['chapter'], categories=chapters_order, ordered=True)
    df_chap = df.groupby('chapter', sort=False)[EMOTION_LABELS_4].mean()
    
    chapter_nums = list(range(1, len(df_chap) + 1))
    ax = axes[i]
    
    for emotion, color in zip(EMOTION_LABELS_4, colors):
        ax.plot(chapter_nums, df_chap[emotion], label=emotion.capitalize(), color=color, alpha=0.7)
    
    ax.set_xticks(chapter_nums)
    ax.set_xticklabels(chapter_nums, ha='right')
    ax.set_title(f'Sentimaent – {titles[i]}')
    ax.set_xlabel('Chapter')
    if i % 4 == 0:
        ax.set_ylabel('Prediction')
    ax.set_ylim(0.125, 0.5)
    ax.grid(True)

axes[7].axis('off')

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=14)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

#%%
# 3. Ansatz
# kapitelweise heatmap für alle emotionen
# emotionen sortieren!

EMOTION_LABELS_WO_NEUTRAL = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
    'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
    'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness',
    'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise'
]

dfs = [HP1_df, HP2_df, HP3_df, HP4_df, HP5_df, HP6_df, HP7_df]
titles = ['HP1', 'HP2', 'HP3', 'HP4', 'HP5', 'HP6', 'HP7']
chapters_all = [HP1_chapters, HP2_chapters, HP3_chapters, HP4_chapters, HP5_chapters, HP6_chapters, HP7_chapters]


chapters_emo_means = {}


fig, axes = plt.subplots(2, 4, figsize=(28, 14), constrained_layout=True)
axes = axes.flatten()

for i, (df, title, chapter_list) in enumerate(zip(dfs, titles, chapters_all)):
    df['chapter'] = pd.Categorical(df['chapter'], categories=chapter_list, ordered=True)
    emo_df = df['emotions'].apply(pd.Series)
    df_full = pd.concat([df[['chapter']], emo_df], axis=1)

    mean_df = df_full.groupby('chapter', sort=False)[EMOTION_LABELS_WO_NEUTRAL].mean().T
    chapters_emo_means[title] = mean_df

    num_chapters = mean_df.shape[1]
    chapter_nums = list(range(1, num_chapters + 1))

    sns.heatmap(mean_df, cmap="RdYlBu_r", linewidths=0.5, annot=False,
                xticklabels=chapter_nums, ax=axes[i])
    axes[i].set_title(f'{title}')
    axes[i].set_xlabel('Chapter')
#    axes[i].set_ylabel('Emotion')
    axes[i].tick_params(axis='x')

axes[-1].axis('off')

plt.show()

#%%

EMOTION_LABELS_WO_NEUTRAL = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
    'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
    'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness',
    'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise'
]

pos_ranked = ['approval','amusement','optimism','pride','relief','gratitude','admiration',
              'caring','excitement','joy','love', 'desire']
neg_ranked = ['annoyance','disappointment','nervousness','remorse','embarrassment',
              'disapproval','fear','sadness','grief','disgust','anger']

ranked_emotions = pos_ranked + neg_ranked

rest_emotions = [em for em in EMOTION_LABELS_WO_NEUTRAL if em not in ranked_emotions]
final_order = ranked_emotions + rest_emotions

fig, axes = plt.subplots(2, 4, figsize=(28, 14), constrained_layout=True)
axes = axes.flatten()

for i, (df, title, chapter_list) in enumerate(zip(dfs, titles, chapters_all)):
    df['chapter'] = pd.Categorical(df['chapter'], categories=chapter_list, ordered=True)
    emo_df = df['emotions'].apply(pd.Series)
    df_full = pd.concat([df[['chapter']], emo_df], axis=1)

    mean_df = df_full.groupby('chapter', sort=False)[EMOTION_LABELS_WO_NEUTRAL].mean().T
    chapters_emo_means[title] = mean_df

    mean_df = mean_df.reindex(final_order)

    num_chapters = mean_df.shape[1]
    chapter_nums = list(range(1, num_chapters + 1))

    sns.heatmap(mean_df, cmap="RdYlBu_r", linewidths=0.5, annot=False,
                xticklabels=chapter_nums, ax=axes[i])
    axes[i].set_title(f'Sentiment – {title}')
    axes[i].set_xlabel('Chapter')
    #axes[i].set_ylabel('Emotion')
    axes[i].tick_params(axis='x')

axes[-1].axis('off')

plt.show()

#%%
# heatmap nur positive

fig, axes = plt.subplots(2, 4, figsize=(24, 12), constrained_layout=True)
axes = axes.flatten()

for i, (df, title, chapter_list) in enumerate(zip(dfs, titles, chapters_all)):
    df['chapter'] = pd.Categorical(df['chapter'], categories=chapter_list, ordered=True)
    emo_df = df['emotions'].apply(pd.Series)
    df_full = pd.concat([df[['chapter']], emo_df], axis=1)

    mean_df = df_full.groupby('chapter', sort=False)[EMOTION_LABELS_WO_NEUTRAL].mean().T
    mean_pos = mean_df.loc[pos_ranked]

    num_chapters = mean_pos.shape[1]
    chapter_nums = list(range(1, num_chapters + 1))

    sns.heatmap(mean_pos, cmap="Greens", linewidths=0.5, annot=False,
                xticklabels=chapter_nums, ax=axes[i])
    axes[i].set_title(f'{title}')
    axes[i].set_xlabel('Chapter')
    #axes[i].set_ylabel('Emotion')
    axes[i].tick_params(axis='x')

axes[-1].axis('off')

plt.show()

#%%

fig, axes = plt.subplots(2, 4, figsize=(24, 12), constrained_layout=True)
axes = axes.flatten()

for i, (df, title, chapter_list) in enumerate(zip(dfs, titles, chapters_all)):
    df['chapter'] = pd.Categorical(df['chapter'], categories=chapter_list, ordered=True)
    emo_df = df['emotions'].apply(pd.Series)
    df_full = pd.concat([df[['chapter']], emo_df], axis=1)

    mean_df = df_full.groupby('chapter', sort=False)[EMOTION_LABELS_WO_NEUTRAL].mean().T
    mean_neg = mean_df.loc[neg_ranked]

    num_chapters = mean_neg.shape[1]
    chapter_nums = list(range(1, num_chapters + 1))

    sns.heatmap(mean_neg, cmap="Reds", linewidths=0.5, annot=False,
                xticklabels=chapter_nums, ax=axes[i])
    axes[i].set_title(f'{title}')
    axes[i].set_xlabel('Chapter')
    #axes[i].set_ylabel('Emotion')
    axes[i].tick_params(axis='x')

axes[-1].axis('off')

plt.show()


#%%
# emotion categorien verteilung

from collections import Counter
from matplotlib.cm import get_cmap


fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
axes = axes.flatten()

cmap = get_cmap('tab20')

for i, (df, title) in enumerate(zip(dfs, titles)):
    ax = axes[i]

    def get_dominant_emotion(emo_dict):
        emo_filtered = {k: v for k, v in emo_dict.items() if k != 'neutral'}
        if not emo_filtered:
            return None
        return max(emo_filtered.items(), key=lambda x: x[1])[0]

    dominant_emotions = df['emotions'].apply(get_dominant_emotion).dropna()
    emotion_counts = Counter(dominant_emotions)
    most_common = emotion_counts.most_common()

    emotions, counts = zip(*most_common)

    colors = [cmap(j % 20) for j in range(len(emotions))]

    ax.barh(emotions, counts, color=colors)
    ax.set_title(f"{title}")
    ax.invert_yaxis() # höchster wert oben
#    ax.set_xlabel()

if len(dfs) < len(axes):
    for i in range(len(dfs), len(axes)):
        axes[i].axis('off')

#plt.suptitle('Count of dominant emotion in sentences', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

#%%


from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

#nochmal nach positiv negativ gruppieren?

EMOTION_LABELS_WO_NEUTRAL = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
    'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
    'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness',
    'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise'
]


pos_ranked = ['approval','amusement','optimism','pride','relief','gratitude','admiration','caring','excitement','joy','love', 'desire']
neg_ranked = ['annoyance','disappointment','nervousness','remorse','embarrassment','disapproval','fear','sadness','grief','disgust','anger']


fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
axes = axes.flatten()
cmap = get_cmap('tab20')

for i, (df, title) in enumerate(zip(dfs, titles)):
    ax = axes[i]

    def get_dominant_emotion(emo_dict):
        emo_filtered = {k: v for k, v in emo_dict.items() if k != 'neutral'}
        if not emo_filtered:
            return None
        return max(emo_filtered.items(), key=lambda x: x[1])[0]

    dominant_emotions = df['emotions'].apply(get_dominant_emotion).dropna()
    emotion_counts = Counter(dominant_emotions)

    emotions_all = list(emotion_counts.keys())

    def emotion_rank(em):
        if em in pos_ranked:
            return pos_ranked.index(em)
        elif em in neg_ranked:
            return len(pos_ranked) + neg_ranked.index(em)
        else:
            return len(pos_ranked) + len(neg_ranked) + emotions_all.index(em)

    emotions_sorted = sorted(emotions_all, key=emotion_rank)

    counts_sorted = [emotion_counts[em] for em in emotions_sorted]
    colors = [cmap(j % 20) for j in range(len(emotions_sorted))]

    ax.barh(emotions_sorted, counts_sorted, color=colors)
    ax.set_title(f"{title}")
    ax.invert_yaxis()

if len(dfs) < len(axes):
    for i in range(len(dfs), len(axes)):
        axes[i].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

#%%


EMOTION_LABELS_WO_NEUTRAL = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
    'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
    'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness',
    'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise'
]


emotion_groups = {
    'positive': ['admiration', 'amusement','approval', 'caring', 'desire', 'excitement', 'gratitude',
                 'joy', 'love', 'optimism', 'pride', 'relief'],
    'negative': ['anger', 'annoyance', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
                 'fear', 'grief', 'nervousness', 'remorse', 'sadness'],
    'ambiguous': ['confusion', 'curiosity', 'realization', 'surprise']
}

fig, axes = plt.subplots(2, 4, figsize=(24, 10), constrained_layout=True)
axes = axes.flatten()

for i, (df, title, chapter_list) in enumerate(zip(dfs, titles, chapters_all)):
    df['chapter'] = pd.Categorical(df['chapter'], categories=chapter_list, ordered=True)

    emo_df = df['emotions'].apply(pd.Series)
    df_full = pd.concat([df[['chapter']], emo_df], axis=1)

    chapter_means = df_full.groupby('chapter', sort=False)[EMOTION_LABELS_WO_NEUTRAL].mean()

    grouped_means = pd.DataFrame(index=chapter_means.index)
    for group_name, group_emotions in emotion_groups.items():
        grouped_means[group_name] = chapter_means[group_emotions].sum(axis=1)

    grouped_means = grouped_means.T

    num_chapters = grouped_means.shape[1]
    chapter_nums = list(range(1, num_chapters + 1))

    sns.heatmap(grouped_means, cmap="YlGnBu", linewidths=0.5, annot=False,
                xticklabels=chapter_nums, ax=axes[i])
    
    axes[i].set_title(f'{title}')
    axes[i].set_xlabel('Chapter')
#    axes[i].set_ylabel('Emotion Type')
    axes[i].tick_params(axis='x', rotation=45)

axes[-1].axis('off')

plt.show()
