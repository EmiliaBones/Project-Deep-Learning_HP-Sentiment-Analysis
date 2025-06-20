import re
import os
import pandas as pd

def chapters_to_dict(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    parts = re.split(r'Chapter\s*:\s*(.+?)\s*\.\|', content) # an chapter teilen
    chapters = {}
    for i in range(1, len(parts), 2):
        title = parts[i].strip()  # 'chapter' entfernen
        text = parts[i + 1].strip()  # dazugehöriges chapter
        sentences = [s.strip() for s in text.split('|') if s.strip()]  # sätze durch | getrennt in liste
        chapters[title] = sentences

    return chapters

# einlesen in dictionary
path = 'HP_books/Harry_Potter_all_char_separated.txt'
chapter_dict = chapters_to_dict(path)

print(chapter_dict.keys())
#print(len(chapter_dict.keys()))

# kapitelüberschriften der bücher
HP1_chapters = ['The Boy Who Lived', 'The Vanashing Glass','The Letters From No One','The Keeper of the Keys','Diagon Alley','The Journey from Platform Nine and Three Quarters','The Sorting Hat','The Potions Master','The Midnight Duel','Halloween','Quidditch','The Mirror of Erised','Nicolas Flamel','Norbert the Norwegian Ridgeback','The Forbidden Forest','Through the Trapdoor','The Man with Two Faces']
HP2_chapters = ['The Worst Birthday','Dobby’s Warning','The Burrow','At Flourish and Blotts','The Whomping Willow','Gilderoy Lockhart','Mudbloods and Murmurs','The Deathday Party','The Writing on the Wall','The Rogue Bludger','The Dueling Club','The Poly Juice Potion','The Very Secret Diary','Cornelius Fudge','Aragog','The Chamber of Secrets','The Heir of Slytherin','Dobby’s Reward']
HP3_chapters = ['Owl Post','Aunt Marge’s Big Mistake','The Knight Bus','The Leaky Cauldron','The Dementor','Talons and Tea Leaves','The Boggart in the Wardrobe','Flight of the Fat Lady','Grim Defeat','The Marauder’s Map','The Firebolt','The Patronus','Gryffindor Versus Ravenclaw','Snape’s Grudge','The Quidditch Final','Professor Trelawney’s Prediction','Cat, Rat and Dog','Moony, Wormtail, Padfoot, and Prongs','The Servant of Lord Voldemort','The Dementor’s Kiss','Hermione’s Secret','Owl Post Again']
HP4_chapters = ['The Riddle House','The Scar','The Invitation','Back to the Burrow','Weasley’s Wizard Wheezes','The Portkey','Bagman and Crouch','The Quidditch World Cup','The Dark Mark','Mayhem at the Ministry','Aboard the Hogwarts Express','The Triwizard Tournament','Mad-Eye Moody','The Unforgivable Curses','Beauxbatons and Durmstrang','The Goblet of Fire','The Four Champions','The Weighing of the Wands','The Hungarian Horntail','The First Task','The House-Elf Liberation Front','The Unexpected Task','The Yule Ball','Rita Skeeter’s Scoop','The Egg and the Eye','The Second Task','Padfoot Returns','The Madness of Mr Crouch','The Dream','The Pensieve','The Third Task','Flesh, Blood, and Bone','The Death Eaters','Priori Incantatem','Veritaserum','The Parting of the Ways','The Beginning']
HP5_chapters = ['Dudley Demented','A Peck of Owls','The Advanced Guard','Number Twelve, Grimmauld Place','The Order of the Phoenix','The Noble and Most Ancient House of Black','The Ministry of Magic','The Hearing','The Woes of Mrs Weasley','Luna Lovegood','The Sorting Hat’s New Song','Professor Umbridge','Detention with Dolores','Percy and Padfoot','The Hogwarts High Inquisitor','In The Hog’s Head','Educational Decree Number Twenty-Four','Dumbledore’s Army','The Lion and the Serpent','Hagrid’s Tale','The Eye of the Snake','St. Mungo’s Hospital for Magical Maladies and Injuries','Christmas on the Closed Ward','Occlumency','The Beetle at Bay','Seen and Unforeseen','The Centaur and the Sneak','Snape’s Worst Memory','Careers Advice','Grawp','O.W.L.s','Out of the Fire']
HP6_chapters = ['The Other Minister','Spinner’s End','Will and Won’t','Horace Slughorn','An Excess of Phlegm','Draco’s Detour','The Slug Club','Snape Victorious','The Half Blood Prince','The House of Gaunt','Hermione’s Helping Hand','Silver and Opals','The Secret Riddle','Felix Felicis','The Unbreakable Vow','A Very Frosty Christmas','A Sluggish Memory','Birthday Surprises','Elf Tails','Lord Voldemort’s Request','The Unknowable Room','After the Burial','Horcruxes','Sectumsempra','The Seer Overheard','The Cave','The Lightning Struck Tower','Flight of the Prince','The Phoenix Lament','The White Tomb']
HP7_chapters = ['The Dark Lord Ascending','In Memoriam','The Dursleys Departing','The Seven Potters','Fallen Warrior','The Ghoul in Pajamas','The Will of Albus Dumbledore','The Wedding','A Place to Hide','Kreacher’s Tale','The Bribe','Magic is Might','The Muggle Born Registration Commission','The Thief','The Goblin’s Revenge','Godric’s Hollow','Bathilda’s Secret','The Life and Lies of Albus Dumbledore','The Silver Doe','Xenophilius Lovegood','The Tale of the Three Brothers','The Deathly Hallows','Malfoy Manor','The Wandmaker','Shell Cottage','Gringotts','The Final Hiding Place','The Missing Mirror','The Lost Diadem','The Sacking of Severus Snape','The Battle of Hogwarts','The Elder Wand','The Prince’s Tale','The Forest Again','King’s Cross','The Flaw in the Plan','Nineteen Years Later']

# alles kapitelüberschriften in uppercase
HP1_chapters = [chapter.upper() for chapter in HP1_chapters]
HP2_chapters = [chapter.upper() for chapter in HP2_chapters]
HP3_chapters = [chapter.upper() for chapter in HP3_chapters]
HP4_chapters = [chapter.upper() for chapter in HP4_chapters]
HP5_chapters = [chapter.upper() for chapter in HP5_chapters]
HP6_chapters = [chapter.upper() for chapter in HP6_chapters]
HP7_chapters = [chapter.upper() for chapter in HP7_chapters]

#print(HP1,HP2,HP3,HP4,HP5,HP6,HP7)

# kapitel in bücher sortieren anhand der kapitelüberschriften in den dicts
HP1 = {key: chapter_dict[key] for key in HP1_chapters if key in chapter_dict}
HP2 = {key: chapter_dict[key] for key in HP2_chapters if key in chapter_dict}
HP3 = {key: chapter_dict[key] for key in HP3_chapters if key in chapter_dict}
HP4 = {key: chapter_dict[key] for key in HP4_chapters if key in chapter_dict}
HP5 = {key: chapter_dict[key] for key in HP5_chapters if key in chapter_dict}
HP6 = {key: chapter_dict[key] for key in HP6_chapters if key in chapter_dict}
HP7 = {key: chapter_dict[key] for key in HP7_chapters if key in chapter_dict}

# anzahl kapitel überprüfen
print(len(HP1_chapters) == len(HP1.keys()))
print(len(HP2_chapters) == len(HP2.keys()))
print(len(HP3_chapters) == len(HP3.keys()))
print(len(HP4_chapters) == len(HP4.keys()))
print(len(HP5_chapters) == len(HP5.keys()))
print(len(HP6_chapters) == len(HP6.keys()))
print(len(HP7_chapters) == len(HP7.keys()))


# dictionarys der bücher in dataframes, 2 spalten: chapter, sentence
def dict_to_df(book_dict):
    data = [
        {"chapter": chapter, "sentence": sentence.strip()}
        for chapter, sentences in book_dict.items()
        for sentence in sentences if sentence.strip()
    ]
    return pd.DataFrame(data)


HP1_df = dict_to_df(HP1)
HP2_df = dict_to_df(HP2)
HP3_df = dict_to_df(HP3)
HP4_df = dict_to_df(HP4)
HP5_df = dict_to_df(HP5)
HP6_df = dict_to_df(HP6)
HP7_df = dict_to_df(HP7)

print(f'Anzahl Sätze alle Bücher: {len(HP1_df)+len(HP2_df)+len(HP3_df)+len(HP4_df)+len(HP5_df)+len(HP6_df)+len(HP7_df)}')

print(HP1_df.head())
print(HP7_df.tail())


print(f'Anzahl Sätze Buch: {len(HP7_df)}')


# abspeichern
# books = [HP1_df,HP2_df,HP3_df,HP4_df,HP5_df,HP6_df,HP7_df]

# def save_books(book_list, output_dir='HP_books'):
#     os.makedirs(output_dir, exist_ok=True)
    
#     for i, df in enumerate(book_list, start=1):
#         filename = f'HP{i}.csv'
#         filepath = os.path.join(output_dir, filename)
#         df.to_csv(filepath, index=False)

# save_books(books)
