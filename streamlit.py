import streamlit as st
import streamlit.components.v1 as stc

# Load
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit

from sklearn.cluster import KMeans
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#Fxn
#Vactorize + Cosine Similarity Matrix
#Recommendation Sys

#load dataset
def load_data(data,error_bad_line = False):
    df = pd.read_csv(data , error_bad_lines = error_bad_line)
    return df

df = load_data("books.csv")

def get_recommendations(data,book_name):
    df = data

    df.loc[(df['average_rating'] >= 0) & (df['average_rating'] <= 1), 'rating_between'] = "between 0 and 1"
    df.loc[(df['average_rating'] > 1) & (df['average_rating'] <= 2), 'rating_between'] = "between 1 and 2"
    df.loc[(df['average_rating'] > 2) & (df['average_rating'] <= 3), 'rating_between'] = "between 2 and 3"
    df.loc[(df['average_rating'] > 3) & (df['average_rating'] <= 4), 'rating_between'] = "between 3 and 4"
    df.loc[(df['average_rating'] > 4) & (df['average_rating'] <= 5), 'rating_between'] = "between 4 and 5"

    # Daha iyi tahmin yapabilme açısından ratingleri,gruplayarak,rating_between içerisine aldım.

    rating_df = pd.get_dummies(df['rating_between'])
    language_df = pd.get_dummies(df['language_code'])  # Makine öğrenmesi modeli geliştirmek için özellik mühendisliği uyguluyorum rating_df ve lanuage_df dataframelerine ilgili değişkenleri one hot encoding uygulayarak tutuyorum.

    features = pd.concat([rating_df,
                          language_df,
                          df['average_rating'],
                          df['ratings_count']],
                         axis=1)  # Encode edilmiş dataframeleri features adında bir dataframe'e concat ediyorum.

    model = neighbors.NearestNeighbors(n_neighbors=6, algorithm='ball_tree')
    model.fit(features)
    dist, idlist = model.kneighbors(features)

    def BookRecommender(book_name):
        book_list_name = []
        book_id = df[df['title'] == book_name].index
        book_id = book_id[0]
        for newid in idlist[book_id]:
            book_list_name.append(df.loc[newid].title)
        return book_list_name

    return BookRecommender(book_name)


def main():
    st.title("Kitap Tavsiye Uygulaması")
    menu = ["Anasayfa","Tavsiye Et","Veri Seti","Hakkında"]
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "Anasayfa":
        st.subheader("Anasayfa")
        st.header("190106109050-FURKAN ORHAN")
        st.header("190106109063-SAMET NİŞANCI")


    elif choice == "Veri Seti":
        dff = load_data("books.csv")
        st.subheader("Veri Seti")
        st.dataframe(dff)

    elif choice == "Tavsiye Et":
        st.subheader("Kitap Tavsiyesi")
        recommend_book = st.text_input("Kitap ismi giriniz.")
        if st.button("Tavsiye Et"):
            if recommend_book is not None:
                result = get_recommendations(df,recommend_book)
                st.write(result)
    else:
        st.subheader("Hakkında")
        st.text("Streamlit & Pandas ile hazırlanmıştır.")

if __name__ == "__main__":
    main()