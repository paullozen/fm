import tw_auth as tw

import pandas as pd
import numpy as np

import plotly
import matplotlib.pyplot as plt

import spacy
from spacy.lang.pt import STOP_WORDS, Portuguese
from collections import Counter
import re
from string import punctuation

from wordcloud import WordCloud

# FUNCTION OF 5 MOST RECENT TWEETS WITH THE HIGHEST NUMBER OF RETWEETS


def extract_data(query, head=5, lang='pt', items=500):
    q_query = ' -filter:retweets'
    TWEET = []
    for tweet in tw.tweepy.Cursor(tw.api.search,
                                  q=query+q_query,
                                  lang=lang,
                                  result_type='recent',
                                  # collect the full text (over 140 characters)
                                  tweet_mode='extended'
                                  ).items(items):
      
        TWEET.append([tweet.full_text.replace('\n', ' '), tweet.retweet_count, [e['text'] for e in tweet._json['entities']['hashtags']]])
        TW = pd.DataFrame(TWEET, columns=['tweet', 'retweets', 'hashtags'])
    return TW


def five_most_recent_highest_retweets(tw, head=5):

    # delete all duplicated texts
    df_five = tw[['tweet', 'retweets']].drop_duplicates()
    # aggregate the same texts and adds up the number of different retweets.
    df_five = df_five.groupby('tweet').sum()
    # put retweets_count in descending order
    df_five = df_five.sort_values(by='retweets', ascending=False).head(head)

    return df_five


# FUNCTION OF # MOST USED AND THEIR RELATIONSHIP


def most_hashtag(df_tweets):
    import pandas as pd
    import numpy as np
    from mlxtend.frequent_patterns import apriori
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import association_rules

    data = df_tweets.hashtags.apply(lambda x: np.nan if len(x) <= 0 else x)
    all_hashtags = list(data.dropna())

    hashtags = []
    for i in all_hashtags:
        for j in i:
            hashtags.append(j)

    hash_str = ''
    for i in hashtags:
        hash_str += i + ' '

        hash_str = hash_str.lower()
        hashtags2 = hash_str.split()

    freq = Counter(hashtags2)
    hash_most_freq = pd.DataFrame(data=freq.most_common(
        10), columns=['hashtag', 'frequency'])
    list_freq = list(hash_most_freq.hashtag)

    all_hashtags_lower = [[h.lower() for h in line] for line in all_hashtags]

    def select_hashtag(freq, all_hash):
        select = []

        for list_hash in all_hash:
            for f in freq:
                if (len(list_hash) >= 2 and (f in list_hash)):
                    select.append(list_hash)
                    break
                else:
                    pass
        return select

    select = select_hashtag(list_freq, all_hashtags_lower)

    te = TransactionEncoder()
    te_ary = te.fit(select).transform(select)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)

    rules = association_rules(
        frequent_itemsets, metric="lift", min_threshold=1)

    return rules


# FUNCTION OF MOST CITED @USER ACCOUNTS IN THE TWEETS


def most_arroba(data):

    tweets = data['tweet']
    arroba = []

    for line in tweets:
        word_split = line.split()
        for word in word_split:
            if word.startswith("@"):
                arroba.append(word)

    top_10 = pd.DataFrame(arroba, columns=['count'])[
        'count'].value_counts().sort_values(ascending=False).head(10)

    return top_10

# FUNCTION OF MOST USED WORDS IN TWEETS DISREGARDING STOPWORDS


def most_words(df_tweets):
    
    def get_all_text(tweet_text):
            txt = ''
            for t in tweet_text:
                txt += t
            return txt

    all_text = get_all_text(df_tweets.tweet).lower()

    # TEXT CLEANING
    ### Special Replacement
    all_text = all_text.replace('inteligencia', 'inteligência')
    all_text = all_text.replace('inteligência artificial', 'ia')
    all_text = all_text.replace('inteligencia artificial', 'ia')
    all_text = all_text.replace('artificial intelligence', 'ia')
    all_text = all_text.replace(punctuation, ' ')
    ###
    sub_text = re.sub(r'http\S+', '', all_text)
    sub_text = re.sub('[-|0-9]',' ', sub_text)
    sub_text = re.findall('\\w+', sub_text)
    sub_text = ' '.join(sub_text)

    # STOPWORDS REMOVAL

    spacy_stopwords = STOP_WORDS
    nlp = Portuguese()

    stopswords_1 = ['pra', 'pro', 'tb', 'tbm', 'vc', 'aí', 'tá', 'ah', 'oq', 'ta'
                    'eh', 'oh', 'msm', 'q', 'r', 'lá', 'ue', 'ué', 'pq', 'ti', 'tu'
                   'rn', 'mt', 'n', 'mais', 'menos', 'pode', 'vai', 'da', 'de',
                   'do', 'uau', 'estao']

    stopwords_2 = ['a','as', 'e', 'es', 'i', 'o', 'os', 'u']

    stopwords_externo = pd.read_csv('portuguese_stopwords.txt', header=None)
    stopwords_3 = stopwords_externo.values.tolist()

    stopwords_4 = []
    for i in stopwords_3:
        stopwords_4.append(i[0])

    stopword_list = set(stopswords_1 + stopwords_2 + stopwords_4)

    spacy_stopwords.update(stopword_list)

    doc = nlp.tokenizer(sub_text)
    words = [token.text for token in doc if token.is_stop != spacy_stopwords]
    final_words = [w for w in words if w not in spacy_stopwords]

    return final_words
