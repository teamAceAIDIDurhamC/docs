import pandas as pd
import numpy as np
import scipy
from scipy.spatial import distance

from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

# Libraries for text preprocessing
import re # remove punctuations, special characters and digits
import nltk

from sklearn.feature_extraction.text import CountVectorizer

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
### Please download these pakages for running this code:
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.svm import SVC
from sklearn.metrics.pairwise import sigmoid_kernel 

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances

def stock_recommend(keyword):
    DATA_URL = ('https://raw.githubusercontent.com/teamAceAIDIDurhamC/docs/master/SP500IT.csv')
    dataset = pd.read_csv(DATA_URL)
    #dataset = pd.read_csv("SP500IT.csv")
    dataset_Org = pd.DataFrame(dataset)
    stop_words=set(stopwords.words("english"))
    cust_input = str(keyword)

    corpus_input= []
    #Remove punctuations
    input_text = re.sub('[^a-zA-Z,]', ' ', cust_input)
    #print(input_text)
    #Convert to lowercase
    input_text = input_text.lower()
    
    #remove tags
    input_text=re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", input_text)

    # remove special characters and digits
    input_text=re.sub("(\\d|\\W)+", " ", input_text)
    
    ##Convert to list from string
    input_text = input_text.split()

    lem = WordNetLemmatizer()
    input_text = [lem.lemmatize(word) for word in input_text if not word in  stop_words] 
    input_text = " ".join(input_text)

    corpus_input.append(str(input_text))
    #print(input_text)
    corpus_input

    cust_input_tokenized = nltk.word_tokenize(input_text) #tokenize customer input after preprocessed
    input_text=cust_input_tokenized

    BoWA = cust_input_tokenized

    Id = int(100000)
    dataset = dataset_Org.append(pd.Series(Id,index=['ID']), ignore_index=True)
    dataset.loc[dataset['ID'] == 100000, ['Name','Description']] = ("customer request", corpus_input)

    corpus= []
    for i in range(0, len(dataset['Description'])):
        #Remove punctuations
        text = re.sub('[^a-zA-Z,]', ' ', str (dataset['Description'][i]))
    
        #Convert to lowercase
        text = text.lower()
    
        #remove tags
        text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
    
        # remove special characters and digits
        text=re.sub("(\\d|\\W)+"," ",text)
    
        ##Convert to list from string
        text = text.split()
    
        ##Stemming
        ps = PorterStemmer()
        
        #Lemmatisation
        lem = WordNetLemmatizer()
        text = [lem.lemmatize(word) for word in text if not word in stop_words] 
        text = " ".join(text)
        corpus.append((text))

    dataset['Processed'] = corpus

    BoWB =[]
    for i in range(0, len(dataset['Processed'])):
        #Remove punctuations
        text_BoW = re.sub('[^a-zA-Z]', ' ', dataset['Processed'][i])
        text_BoW = text_BoW.split()
        BoWB.append((text_BoW))

    dataset['BoWB'] = BoWB

    cv = CountVectorizer(max_df=0.8, stop_words=stop_words, max_features=10000, ngram_range=(1,1))
    X = cv.fit_transform(dataset['Processed'])

    dataset['dicB'] = X

    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(dataset['Processed'])

    tf.fit(dataset['Processed'])

    cosine_similarities = sigmoid_kernel(tfidf_matrix, tfidf_matrix) 

    similarities = cosine_similarities 

    results = {}
    for idx, row in dataset.iterrows():
        similar_indices = similarities[idx].argsort()[:-100:-1] 
        similar_items = [(similarities[idx][i], dataset['ID'][i]) for i in similar_indices] 
        results[row['ID']] = similar_items[1:]
    
    def item(id):
        return dataset.loc[dataset['ID'] == id]['Name'].tolist()[0].split(' - ')[0]

    def recommend(item_id, num):
        #print("Recommending " + str(num) + " products similar to " + item(item_id) + "...")
        #print("-------")
        recs = results[item_id][:num]
        rec_list = pd.DataFrame(recs, columns=['Matching Score', 'Company Name'])
        
        #for rec in recs:
            #print("Recommended: " + item(rec[1]) + " (Matching score:" + str("%.5f" %rec[0]) + ")")
            # print(rec)
        for i in range(num):
            rec_list['Company Name'][i] = item(rec_list['Company Name'][i])
        
        rec_list = rec_list[['Company Name', 'Matching Score']]
        #print(rec_list)
        return rec_list
    # recommend(item_id=100000, num=100)
    return recommend(item_id=100000, num=5)

    # recommend(item_id=100000, num=100)
