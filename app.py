import plotly.express as px
import streamlit as st
from functions import *
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter

from PIL import Image
from pathlib import Path  # para a logo
import base64  # para a logo


# LOGO

# Main


def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


    # Extração de tweets com a plavra input

# Side bar

def main():

    sidebar_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
        img_to_bytes("logo2.png"))


    st.sidebar.markdown(sidebar_html, unsafe_allow_html=True)

    # FIM LOGO

    """
    # Fatal Model - Twitter

    Bem-vindo(a). Aqui você pode ter algumas análises de tweets de acordo com o termo pesquisado na barra lateral esquerda.
    """

    # # SIDE BAR

    # st.sidebar.markdown(" ")
    # st.sidebar.markdown("**• [Documentação](https://bit.ly/39YuSuv)**")

    st.sidebar.markdown('**Defina a quantidade de tweets que deseja pesquisar:**')
    qnt_input = st.sidebar.slider('', 500, 2000)

    # word input
    st.sidebar.markdown('**Defina o que deseja pesquisar:**')
    input_word = st.sidebar.text_input('')

    # checks
    st.sidebar.markdown(""" **Selecione as opções de busca:** """)
    check1 = st.sidebar.checkbox('Os 5 tweets mais retweetados')
    check2 = st.sidebar.checkbox('Os 10 usuários mais citados')
    check3 = st.sidebar.checkbox('As palavras mais usadas')
    check4 = st.sidebar.checkbox('As hashtags mais usadas e suas relações')
    st.sidebar.markdown('----')

    # FIM SIDE BAR
    
    # FUNCTIONS
    
    @st.cache
    def load_data(p_input):
        df = extract_data(p_input, items=qnt_input)
        return df
    
    if input_word != '':
        df_tweets = load_data(input_word)
        st.sidebar.markdown(len(input_word))
    
        #######################################################################
    
        # 5 TWEETS MAIS RETWEETADOS
        st.markdown(f'Você selecionou um range de busca de **{qnt_input}** tweets.')
        if check1:
    
            """ ### Os 5 Tweets Mais Retweetados:"""
            tweets_5 = five_most_recent_highest_retweets(
                df_tweets)  # Chamada da função
    
            # Mostra o resultado dos 5 tweets na tela
    
            st.table(tweets_5)
    
        #######################################################################
    
        # @ MAIS CITADOS
    
        if check2:
    
            users = most_arroba(df_tweets)  # chamada da função
    
            """ ### @ Usuários Mais Citados """
    
            plot_users = px.bar(users, y=users.index, x='count',
                                text='count', labels={})
            plot_users['layout']['yaxis']['autorange'] = "reversed"
    
            st.plotly_chart(plot_users)
    
        #######################################################################
    
        # PALAVRAS MAIS USADAS
    
        if check3:
    
            words = most_words(df_tweets)  # chamada da funçao
    
            freq_all_words = Counter(words)
            freq_df = pd.DataFrame(data=freq_all_words.most_common(
                10), columns=['Palavras', 'Frequências'])[1:]
    
            # Plota as palavras mais frequentes
    
            """ ### As Palavras Mais Usadas """
            plot_freq = px.bar(freq_df, y='Palavras', x='Frequências',
                            orientation='h', text='Frequências')
            plot_freq['layout']['yaxis']['autorange'] = "reversed"
    
            st.plotly_chart(plot_freq)
        
            st.markdown(f'\nThe search term ({input_word.upper()}) has returned {freq_all_words.most_common(1)[0][1]} times.')

    
            # Plota a nuvem de palavras
            """ ### Nuvem de Palavras
            Quanto maior a frequência da palavra, maior ela se apresenta na nuvem """
    
            twitter_fig = np.array(Image.open("twitter_black.png"))
    
            words_str = ' '.join(words)  # word list into a string
    
            wordcloud = WordCloud(max_font_size=100, width=1520,
                                height=535, max_words=100,
                                mask=twitter_fig, background_color='white').generate(words_str)
            plt.figure(figsize=(8, 7))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            st.pyplot()
               
        #######################################################################
    
        # ASSOCIAÇÃO DE HASHTAGS
    
        if check4:
    
            rules = most_hashtag(df_tweets)  # Chamada da função
    
            rules.antecedents = rules.antecedents.apply(lambda x: next(iter(x)))
            rules.consequents = rules.consequents.apply(lambda x: next(iter(x)))
    
            """ ### Associação entre #hashtags
            Caso não apareça, é porque não há associação."""
            fig, ax = plt.subplots(figsize=(16, 9))
            GA = nx.from_pandas_edgelist(
                rules, source='antecedents', target='consequents')
            circPos=nx.circular_layout(GA)
    
            pos_attrs = {}
    
            for node, coords in circPos.items():
                pos_attrs[node] = (coords[0]+0.1*(-1)*np.sign(coords[0]),
                                coords[1]+0.1*(-1)*np.sign(coords[1]))
    
            nx.draw(GA, with_labels=True, pos=pos_attrs, alpha=0.3)
            nx.draw_networkx_labels(GA, pos_attrs, alpha=1)
    
            st.pyplot()
    
        #######################################################################
    
    else:
        """ ### Por favor, insira uma palavra no campo de busca na barra ao lado """


if __name__ == '__main__':
    main()
