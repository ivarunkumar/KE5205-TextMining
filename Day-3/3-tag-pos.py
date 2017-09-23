# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 00:19:41 2017
Workshop: IE - POS Tagging
@author: issfz
"""

import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import reuters
from nltk.corpus import stopwords
import string

stop = stopwords.words('english')
# ===== POS Tagging using NLTK =====

sent = '''Professor Tan Eng Chye, NUS Deputy President and Provost, and Professor 
Menahem Ben-Sasson, President of HUJ signed the joint degree agreement at NUS, 
in the presence of Ambassador of Israel to Singapore Her Excellency Amira Arnon 
and about 30 invited guests, on July 03, 2013.
'''

# The input for POS tagger needs to be tokenized first.
sent_pos = pos_tag(word_tokenize(sent))
sent_pos

# A more simplified tagset - universal
sent_pos2 = pos_tag(word_tokenize(sent), tagset='universal')
sent_pos2


# The wordnet lemmatizer works properly with the pos given
wnl = nltk.WordNetLemmatizer()


def process(toks) :
    toks = [t for t in toks if t not in stop ]
    toks = [ t.lower() for t in toks if t not in string.punctuation ]
    toks = [ t for t in toks if len(t) >= 3 ]
    toks = pos_tag(word_tokenize(toks), tagset='universal')
     # adjectives and adverbs
    tokens_non_lem = [ t[0] for t in toks  if t[1] == "ADJ" or t[1] == "ADV" ]
    # nouns and verbs
    noun_tokens_lem = [ wnl.lemmatize(t[0], pos = 'n') for t in toks  if t[1] == "NOUN" ]
    verb_tokens_lem = [ wnl.lemmatize(t[0], pos = 'v') for t in toks  if t[1] == "VERB" ]
    tokens_clean = tokens_non_lem + noun_tokens_lem + verb_tokens_lem
    return tokens_clean

# Let's pick two categories and visualize the articles in each category using word cloud
grain = reuters.fileids('grain')
trade = reuters.fileids('trade')
reuters.words(grain[0])

grain_tok = [ reuters.words(f) for f in grain ] 
trade_tok = [ reuters.words(f) for f in trade ] 

# Preprocess each file in each category
grain_clean = [ process(f) for f in grain_tok ]
trade_clean = [ process(f) for f in trade_tok ]

# Flatten the list of lists for FreqDist
grain_flat = [ c for l in grain_clean for c in l ]
trade_flat = [ c for l in trade_clean for c in l ]

fd_grain = FreqDist(grain_flat)
fd_trade = FreqDist(trade_flat)

# Generate word clouds for the two categories.
from wordcloud import WordCloud
import matplotlib.pyplot as plt

wc_grain = WordCloud(background_color="white").generate_from_frequencies(fd_grain)
plt.imshow(wc_grain, interpolation='bilinear')
plt.axis("off")
plt.show()

wc_trade = WordCloud(background_color="white").generate_from_frequencies(fd_trade)
plt.imshow(wc_trade, interpolation='bilinear')
plt.axis("off")
plt.show()

fd_clean = nltk.FreqDist(tokens_clean)
# Join the cleaned tokens back into a string.
# Why? Because some functions we'll use later require string as input.
text_clean=" ".join(tokens_clean)
#------------------------------------------------------------------------
# Exercise: remember the wordcloud we created last week? Now try creating 
# a wordcloud with only nouns, verbs, adjectives, and adverbs, with nouns 
# and verbs lemmatized.
#-------------------------------------------------------------------------


# Generate word clouds for the two categories.
'''
from wordcloud import WordCloud
import matplotlib.pyplot as plt

wc_grain = WordCloud(background_color="white").generate_from_frequencies(fd_clean)
plt.imshow(wc_grain, interpolation='bilinear')
plt.axis("off")
plt.show()
'''