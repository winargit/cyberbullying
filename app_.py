from keras import backend as K
from tensorflow.keras.models import Model, load_model
import streamlit as st
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


MODEL_PATH = r"models/LSTM_model_1.h5"
MAX_NB_WORDS = 100000 # max no. of words for tokenizer
MAX_SEQUENCE_LENGTH = 200 # max length of each entry (sentence), including padding
VALIDATION_SPLIT = 0.2 # data for validation (not used in training)
EMBEDDING_DIM = 100
jenismodel=1

def getModel(idModel):
    modelname = ['Naive Bayes','Log Regression','Support Vector Classifier']
    modellist = ['model_NB.pkl','model_logreg.pkl','model_svc.pkl']
    return modelname[idModel], modellist[idModel]

tokenizer_file = "vectorizer.pkl"
wordnet = WordNetLemmatizer()
regex = re.compile('[%s]' % re.escape(string.punctuation))
model_list = ["Logistic Regression","Naive Bayes","SVM"]
model_file_list = [r"./model_logreg.pkl",r"model_NB.pkl",r"model_svc.pkl"]

with open(tokenizer_file, 'rb') as handle:
   tokenizer = pickle.load(handle)

def basic_text_cleaning(line_from_column):
    tokenized_doc = word_tokenize(line_from_column)
    return tokenized_doc

def get_data():
    return []


@st.cache(allow_output_mutation=True)
def Load_model():
    model=open('model_logreg.pkl', 'rb')
  # model._make_predict_function()
  # model.summary() # included to make it visible when model is reloaded
   # session = K.get_session(op_input_list=())
    return model


if __name__ == '__main__':
   ambilmodel=getModel(1); 
   #print(ambilmodel[0])
   #print('Algoritma {}'.format(ambilmodel[0]))
   st.title('Prediksi Twitter Cyberbulying  ')
   st.info('Learning Model Menggunakan Algoritma {}'.format(ambilmodel[0],"s"))
   st.subheader('Masukkan dalam teks boks di bawah ini')
   option = st.selectbox(
    'Masukkan Algoritma yang ingin dipilih',["Logistic Regression","Naive Bayes","SVC"])
   sentence = st.text_area('Masukkan konten Twit di sini', 'Some news',height=200)
   predict_btt = st.button('Prediksi')
   
 

if predict_btt:
   clean_text = []
   #print(sentence)
   #model = Load_model()
   print('Opsi {}'.format(option))
   def process_tweet(sentence):
    return " ".join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])", " ",sentence.lower()).split())

   datainput={"text":[sentence]}
   df_test=pd.DataFrame(datainput)
   df_test['processed_text'] = df_test['text'].apply(process_tweet)
   df_test.head()
   X = df_test['processed_text']
   # x_test_counts = count_vect.transform(X)
   x_test_tfidf = tokenizer.transform(X)
   pickle_in = open(getModel(jenismodel)[1],'rb')
   model=pickle.load(pickle_in)
   #i = basic_text_cleaning(sentence)
   #clean_text.append(i)
   #print(clean_text.shape)
   #sequences = tokenizer.transform(clean_text)
   # model.fit(X_train_tf,y_train)
   #y_pred = log_reg.predict(X_test_tf)
   #data = pad_sequences(sequences, padding = 'post', maxlen =  MAX_SEQUENCE_LENGTH)
   #decission={ 1:"religion",2:"age", 3:"ethnicity", "gender": 4, "other_cyberbullying": 5,"not_cyberbullying": 6}
   class_name=["","religion","age","ethnicity", "gender", "other_cyberbullying","not_cyberbullying"]
   prediction = model.predict(x_test_tfidf)
   index_pred = round( (np.max(prediction)),5)
   #print(np.argmax(prediction))
   final_pred = class_name[index_pred]
   st.success('Algoritma yang digunakan {}'.format(getModel(jenismodel)[0]))
   st.success('Hasil Prediksi Twitter adalah: {}'.format(final_pred))
   #st.write('Tingkat Kepercayaan : {}%'.format(prediction))