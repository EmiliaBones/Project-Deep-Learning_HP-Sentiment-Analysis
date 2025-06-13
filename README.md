This was a final project of the Deep Learning course as part of the Machine Learning Developer training in May 2025.

The aim was a sentiment analysis of the Harry Potter books to compare the color plots of the Harry Potter films (see presentation pdf). 
Hypothesis: The development of the mood in the books is becoming increasingly negative, as suggested by the colors of the scenes in the Harry Potter films.

First, the data was prepared: 7 books, by chapter, one sentence per line.

For a rough first overview, a simple LSTM model was first trained with a Twitter dataset labeled for two sentiments (positive and negative) and applied to the 7 datasets and averaged for each chapter.

A fine-tuned model from huggingface (https://huggingface.co/bhadresh-savani/bert-base-go-emotion) was then used, which consists of the base model distilbert-base-uncased and a classifier layer for 28 classes. Training was carried out with the GoEmotions data set from Google, which codes 27 emotions and neutral moods (https://arxiv.org/pdf/2005.00547v2).

The results are then visualized. No particular trend in mood development can be identified. The most frequent sentiments are almost the same across books and the original GoEmotions dataset. Since the sentence-by-sentence classification means that no context can be taken into account, the data set has now been prepared in such a way that larger text sections of 10 sentences are evaluated.

These results are presented using line plots and heat maps. Different moods crystallize than before, and a mood progression can also be seen, although it is still not as clear as that of the film color plots.
