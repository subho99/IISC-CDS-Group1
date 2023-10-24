import streamlit as st
#from pyspark.sql import SparkSession
#from pyspark.sql.functions import *
import re
import string
#from pyspark.ml.feature import VectorAssembler
#from pyspark.ml.feature import CountVectorizer
#from pyspark.ml.classification import NaiveBayes
#from pyspark.sql.types import ArrayType, StringType
#from handyspark import *

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
#from pyspark.sql.functions import udf
#import handyspark as hsp
import matplotlib.pyplot as plt
#from pyspark.sql.types import IntegerType
from nltk.corpus import wordnet

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from matplotlib.gridspec import GridSpec
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
import string
from wordcloud import WordCloud
import wordcloud
import re
import string
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import lightgbm as lgb

#Custom import
from pages.scripts.content.util import get_base_path, get_content_folder, get_content_path
from pages.scripts.content.util import get_content_pkl_path, get_classified_lable_file_path, is_debug

#from google.colab import drive  AK
#------------------------------------------------------

#drive.mount('/content/drive', force_remount=True)  AK
#https://colab.research.google.com/github/mrm8488/shared_colab_notebooks/blob/master/Create_streamlit_app.ipynb
#https://discuss.streamlit.io/t/how-to-launch-streamlit-app-from-google-colab-notebook/42399
#------------------------------------------------------

import pandas as pd


# Define the path to the Excel file
excel_file_path = 'pages/scripts/content/STG_excelfile.xlsx'

# Read the Excel file into a DataFrame
df = pd.read_excel(excel_file_path)

# Define the path to save the CSV file
csv_file_path = 'pages/scripts/content/STG_excelfile.csv'

# Save the DataFrame as a CSV file
df.to_csv(csv_file_path, index=False)

print("Excel to CSV conversion completed successfully.")
#------------------------------------------------------

df = pd.read_csv(csv_file_path)
print(df.shape)
df.head(10)      # first 10 rows
#------------------------------------------------------

# Creating a new column cleaned_resume
df['cleaned_sentence'] = ''
df.head()
#------------------------------------------------------

# Extracting the information regarding the non-null counts of each of the features and target
df.info()
#------------------------------------------------------

# Display the distinct categories of resume
# YOUR CODE
import numpy as np
print("-----Action label Categories-----")
print("Displaying the distinct categories of action labels:\n ")
#print(df['Category'].unique())
# Iterate through each row in the DataFrame
action_labels = []
for index, row in df.iterrows():
    # Extract 'SentenceID' and split to get the action label
    sentence_id = row['SentenceID']
    action_label = sentence_id.split('_')[2]
    action_labels.append(action_label)
print(np.unique(action_labels))
#------------------------------------------------------

df['labels'] = ''
df.head()
#------------------------------------------------------

df['labels'] = action_labels
df.head()
#------------------------------------------------------

# Applying str.lower() directly on top of the pandas dataframe column 'Resume'
print("Before converting the column Sentence to lowercase\n")
print(df['Sentence'].head(7))
df['Sentence']=df['Sentence'].str.lower()
print("After converting the column Sentence to lowercase\n")
print(df['Sentence'].head(7))
#------------------------------------------------------

import wordcloud
import re
import string
def cleanSentence(SentenceText):
    # YOUR CODE HERE
    cleantext = ' '.join(re.sub('(#[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+\s+)'," ",SentenceText).split()) #Hashtags, # and Mentions, @,URL,White space
    cleantext = ' '.join(re.sub('RT|cc'," ",cleantext).split()) # remove rt and cc
    cleantext=' '.join(word.strip(string.punctuation) for word in cleantext.split())# remove punctuations
    return cleantext
print("Before Cleaning the text in column 'Sentence'\n", df['Sentence'].head())
df['cleaned_sentence'] = df.Sentence.apply(lambda x: cleanSentence(x))
print("\nAfter Cleaning the text in column 'Sentence'\n", df['cleaned_sentence'].head())
#------------------------------------------------------

sent_lens = []
for i in df.cleaned_sentence:
    length = len(i.split())
    sent_lens.append(length)
# all lines below commented by AK
# print("No. of words in each sentence:")
# print(sent_lens,'\n')
# print("Total no. of sentences submitted:")
# print(len(sent_lens), '\n')
# print("Wordcount in the highest length sentence:")
# print(max(sent_lens))
#------------------------------------------------------

# extracting the stopwords in english language
English_stopwords = nltk.corpus.stopwords.words('english')
print("English stopwords: ",English_stopwords)
#------------------------------------------------------

# most common words
# YOUR CODE HERE
from nltk.tokenize import word_tokenize
import string
punct_ch = string.punctuation

# removing the stopwords and punctuations
def remove_stopwords_and_punctuation(text, is_lower_case=False):
    # Step-1:
    # tokenizing the complete text from a resume and generating a list of unique words (or tokens)
    tokens_list = word_tokenize(text)
    # print("List of tokens in each resume: \n")
    # print(tokens_list)

    # Step-2:
    # Using list comprehension and The Strip() method to remove or truncate the given characters from the beginning and the end
    # of the original string. The default behavior of the strip() method is to remove the whitespace from the beginning and at the end
    # of the string.
    word_tokens = [word_token.strip() for word_token in tokens_list]
    # print("List of stripped word tokens in each resume: \n")
    # print(word_tokens)

    # Step-3:
    # Filtering out the stop words and if any remaining punctuations from the list of 'word_tokens'
    filtered_tokens = [token for token in word_tokens if token not in English_stopwords and token not in punct_ch]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text
#------------------------------------------------------

# Step-4:
# Finally applying the 'remove_stopwords_and_punctuation' function on the 'cleaned_resume' column
df['cleaned_sentence'] = df['cleaned_sentence'].apply(remove_stopwords_and_punctuation)
df.head(7)
#------------------------------------------------------

for resume in df.cleaned_sentence[0:3]:
    print(resume)
#------------------------------------------------------

AllsentencesWords_combinedText = ' '.join(words_per_sentence for words_per_sentence in df.cleaned_sentence[0:3])
print(AllsentencesWords_combinedText)
#------------------------------------------------------

# YOUR CODE HERE to show the most common word using WordCloud
# Option-2: Using WordCloud
AllsentencesWords_combinedText = ' '.join(words_per_sentence for words_per_sentence in df.cleaned_sentence)
WC = WordCloud(width=1000, height=500, max_words=500, min_font_size=5,collocations=False)
TotalWords = WC.generate(AllsentencesWords_combinedText)
plt.figure(figsize=(10,10))
plt.imshow(TotalWords, interpolation='bilinear')
plt.axis("on")
plt.show()
#------------------------------------------------------

df['labels']
#------------------------------------------------------

from sklearn.preprocessing import LabelEncoder
import pandas as pd
# Copying the dataframe 'df' to 'df_final'
df_final = df.copy()
# Creating a new column LabelEncoded_Category
df_final['LabelEncoded_Action_labels'] = ''
df_final.head()
#------------------------------------------------------

# Creating Label Encoder object
le = LabelEncoder()
# Using .fit_transform function to fit the labelencoder and it encodes each of the string in 'Category' column
# and converts each string into a label encoded integer value being stored under the new column 'LabelEncoded_Category'
df_final['LabelEncoded_Action_labels'] = le.fit_transform(df_final['labels'])
df_final.head(7)
#------------------------------------------------------

# YOUR CODE HERE
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

requiredFeature = df_final['cleaned_sentence'].values
# print("Cleaned Text: \n"
print("requiredFeature:\n", requiredFeature)
requiredTarget = df_final['LabelEncoded_Action_labels'].values

Tfidf_word_vectorizer = TfidfVectorizer(ngram_range = (1,3)) # sublinear_tf=True, stop_words='english'
X_tfidf = Tfidf_word_vectorizer.fit_transform(requiredFeature)

if is_debug() == True:
    print ('Tfidf_train:', X_tfidf.shape)
    print ("Text converted to Tfidf Feature vectors shown below:\n")
    print(X_tfidf)
#------------------------------------------------------

# YOUR CODE HERE
X_train,X_test,y_train,y_test = train_test_split(X_tfidf, requiredTarget, random_state=0, test_size=0.2, shuffle=True, stratify=requiredTarget)
print (X_train.shape)
print (X_test.shape)
print (y_train.shape)
print (y_test.shape)
#------------------------------------------------------

import joblib
import os

base_path = get_base_path() # /user/home
content_folder = get_content_folder()  # content
content_path = get_content_path() # /user/home/content
pkl_folder_path = get_content_pkl_path() # /user/home/content


#st.write(content_path)
# best_rf_classifier_file = os.path.join(pkl_folder_path, 'best_rf_classifier.pkl')
# loaded_xgb_classifier_file = os.path.join(pkl_folder_path, 'best_xgb_classifier.pkl')
# loaded_lgb_classifier_file = os.path.join(pkl_folder_path, 'best_lgb_classifier.pkl')
#classified_label_file = os.path.join(content_path, 'classified_label.txt')

# if is_debug() == True:
#     print(" best_rf_classifier_file path -> ", best_rf_classifier_file)
#     print(" loaded_xgb_classifier_file path -> ", loaded_xgb_classifier_file)
#     print(" loaded_lgb_classifier_file path ->", loaded_lgb_classifier_file)
#     print(" classified_label_file path ->", get_classified_lable_file_path())


if is_debug()==True:
    print("Loading the 'best_rf_classifier'..")

# # Load the saved model
# loaded_rf_classifier = joblib.load(best_rf_classifier_file)

#------------------------------------------------------

if is_debug()==True:
    print("Loading the 'best_xgb_classifier'..")

# import joblib

# # Load the saved model
# loaded_xgb_classifier = joblib.load(loaded_xgb_classifier_file)
#------------------------------------------------------

if is_debug()==True:
    print("Loading the 'best_lgb_classifier'..")

# import joblib

# # Load the saved model
# loaded_lgb_classifier = joblib.load(loaded_lgb_classifier_file)
#------------------------------------------------------

def preprocess_text(text):
    text = cleanSentence(text)
    text = remove_stopwords_and_punctuation(text)
    text = text.lower()
    return text
#------------------------------------------------------
def load_rf_model():
        # base_path = get_base_path() # /user/home
        # content_folder = get_content_folder()  # content
        # content_path = get_content_path() # /user/home/content
        pkl_folder_path = get_content_pkl_path() # /user/home/content
        best_rf_classifier_file = os.path.join(pkl_folder_path, 'best_rf_classifier.pkl')
        # Load the saved model
        loaded_rf_classifier = joblib.load(best_rf_classifier_file)
        return loaded_rf_classifier

def load_xgb_model():
        # base_path = get_base_path() # /user/home
        # content_folder = get_content_folder()  # content
        # content_path = get_content_path() # /user/home/content
        pkl_folder_path = get_content_pkl_path() # /user/home/content
        loaded_xgb_classifier_file = os.path.join(pkl_folder_path, 'best_xgb_classifier.pkl')
        # Load the saved model
        loaded_xgb_classifier = joblib.load(loaded_xgb_classifier_file)
        return loaded_xgb_classifier


def load_lgb_model():
        # base_path = get_base_path() # /user/home
        # content_folder = get_content_folder()  # content
        # content_path = get_content_path() # /user/home/content
        pkl_folder_path = get_content_pkl_path() # /user/home/content
        loaded_lgb_classifier_file = os.path.join(pkl_folder_path, 'best_lgb_classifier.pkl')
        # Load the saved model
        loaded_lgb_classifier = joblib.load(loaded_lgb_classifier_file)
        return loaded_lgb_classifier


def predict_resume_with_rf(text):
    loaded_rf_classifier = load_rf_model()
    processed_text = preprocess_text(text)
    review = Tfidf_word_vectorizer.transform([processed_text])
    pred = loaded_rf_classifier.predict(review)
    outputCategory = df_final.labels[df_final['LabelEncoded_Action_labels']==pred[0]].unique()
    with open(get_classified_lable_file_path(), 'w') as f:
        f.write(outputCategory[0])
    return outputCategory
#------------------------------------------------------

def predict_resume_with_xgb(text):
  loaded_xgb_classifier = load_xgb_model()
  processed_text = preprocess_text(text)
  review = Tfidf_word_vectorizer.transform([processed_text])
  pred = loaded_xgb_classifier.predict(review)
  outputCategory = df_final.labels[df_final['LabelEncoded_Action_labels']==pred[0]].unique()
  with open(get_classified_lable_file_path(), 'w') as f:
    f.write(outputCategory[0])
  return outputCategory
#------------------------------------------------------

def predict_resume_with_lgb(text):
  loaded_lgb_classifier = load_lgb_model()
  processed_text = preprocess_text(text)
  review = Tfidf_word_vectorizer.transform([processed_text])
  pred = loaded_lgb_classifier.predict(review)
  outputCategory = df_final.labels[df_final['LabelEncoded_Action_labels']==pred[0]].unique()
  with open(get_classified_lable_file_path(), 'w') as f:
    f.write(outputCategory[0])
  return outputCategory
#------------------------------------------------------

#classified_label=""
# user_input = st.text_input("Enter the text input","Your tweet here ")   #AK
# if st.button('classify with RandomForest'):
#     # Predicting the unknown action label
#     pred_action_label_rf = predict_resume_with_rf(user_input)
#     classified_label = pred_action_label_rf[0]
#     st.write(pred_action_label_rf[0])

# elif st.button('classify with XGBoost'):
#     # Predicting the unknown action label
#     pred_action_label_xgb = predict_resume_with_xgb(user_input)
#     classified_label = pred_action_label_xgb[0]
#     st.write(pred_action_label_xgb[0])

# elif st.button('classify with Microsoft LGBM'):
#     # Predicting the unknown action label
#     pred_action_label_lgb = predict_resume_with_lgb(user_input)
#     classified_label = pred_action_label_lgb[0]
#     st.write(pred_action_label_lgb[0])

#with open(os.path.join(content_path, 'classified_label.txt'), 'w') as f:
#   f.write(classified_label)
