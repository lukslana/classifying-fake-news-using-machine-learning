import streamlit as st
import requests
import pandas as pd
import os
import subprocess
import sys

try:
    from bs4 import BeautifulSoup
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'beautifulsoup4'])
    from bs4 import BeautifulSoup

#from sklearn.feature_extraction.text import CountVectorizer
try:
    from sklearn.feature_extraction.text import CountVectorizer
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'scikit-learn'])
    from sklearn.feature_extraction.text import CountVectorizer

try:
    import matplotlib.pyplot as plt
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'matplotlib'])
    import matplotlib.pyplot as plt

def fetch_news_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        # G1 news page structure
        title_tag = soup.find('h1')
        title = title_tag.get_text(strip=True) if title_tag else ''
        # Try to get the main text
        fulltext = ''
        article = soup.find('div', {'class': 'mc-article-body'})
        if article:
            paragraphs = article.find_all('p')
            fulltext = '\n'.join([p.get_text(strip=True) for p in paragraphs])
        else:
            # fallback: get all <p> tags
            paragraphs = soup.find_all('p')
            fulltext = '\n'.join([p.get_text(strip=True) for p in paragraphs])
        df = pd.DataFrame([{'title': title, 'fulltext': fulltext, 'link': url}])
        return df
    except Exception as e:
        st.error(f'Erro ao buscar a notícia: {e}')
        return None

def save_to_csv(data, csv_path):
    if os.path.exists(csv_path):
        data.to_csv(csv_path, mode='a', header=False, index=False, encoding='utf-8')
    else:
        data.to_csv(csv_path, mode='w', header=True, index=False, encoding='utf-8')

st.title('Analisar Notícias')
url = st.text_input('Cole a URL da notícia:')
if st.button('Buscar e Gravar'):
    if url:
        news_df = fetch_news_data(url)
        if news_df is not None:
            save_to_csv(news_df, 'src/data/news-dataset.csv')
            st.success('Notícia gravada com sucesso!')
            st.write('**Título:**', news_df.iloc[0]["title"])
            st.write('**Link:**', news_df.iloc[0]["link"])
            st.write('**Texto:**')
            st.write(news_df.iloc[0]["fulltext"])

            # Bag of Words do artigo recém-adicionado
            st.subheader('Bag of Words deste artigo')
            stopwords_pt = [
                'a', 'à', 'ao', 'aos', 'as', 'àquela', 'àquelas', 'àquele', 'àqueles', 'àquilo', 'com', 'como', 'da', 'das', 'de', 'dela', 'delas', 'dele', 'deles', 'depois', 'do', 'dos', 'e', 'é', 'ela', 'elas', 'ele', 'eles', 'em', 'entre', 'era', 'eram', 'essa', 'essas', 'esse', 'esses', 'esta', 'está', 'estão', 'estas', 'estava', 'estavam', 'este', 'estes', 'eu', 'foi', 'foram', 'há', 'isso', 'isto', 'já', 'lhe', 'lhes', 'mais', 'mas', 'me', 'mesmo', 'meu', 'meus', 'minha', 'minhas', 'na', 'nas', 'não', 'nem', 'no', 'nos', 'nós', 'nossa', 'nossas', 'nosso', 'nossos', 'num', 'numa', 'o', 'os', 'ou', 'para', 'pela', 'pelas', 'pelo', 'pelos', 'por', 'qual', 'quando', 'que', 'quem', 'se', 'sem', 'ser', 'seu', 'seus', 'só', 'sua', 'suas', 'também', 'te', 'tem', 'tendo', 'tenho', 'ter', 'teu', 'teus', 'teve', 'tinha', 'tinham', 'tive', 'tu', 'tua', 'tuas', 'um', 'uma', 'você', 'vocês'
            ]
            col1, col2 = st.columns(2)

            artigo_text = news_df.iloc[0]["fulltext"]
            vectorizer_artigo = CountVectorizer(stop_words=stopwords_pt, max_features=20)
            X_artigo = vectorizer_artigo.fit_transform([artigo_text])
            bow_artigo = dict(zip(vectorizer_artigo.get_feature_names_out(), X_artigo.toarray()[0]))
            bow_df_artigo = pd.DataFrame(list(bow_artigo.items()), columns=['Palavra', 'Frequência']).sort_values('Frequência', ascending=False)

            with col1:
                st.markdown('**Tabela Bag of Words**')
                st.dataframe(bow_df_artigo)

            # Exibir apenas as 10 palavras mais frequentes para melhor visualização
            import altair as alt
            top_bow_df = bow_df_artigo.head(10)
            with col2:
                st.markdown('**Gráfico das 10 palavras mais frequentes deste artigo:**')
                chart = alt.Chart(top_bow_df).mark_bar(color='#1976d2').encode(
                    x=alt.X('Palavra', sort='-y', title='Palavra'),
                    y=alt.Y('Frequência', title='Frequência')
                ).properties(width=500, height=300)
                st.altair_chart(chart, use_container_width=True)
    else:
        st.warning('Por favor, insira uma URL.')

