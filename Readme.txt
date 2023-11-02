Sentiment Analysis  of  
Marketing 
  
 Abstract  
  
  
Sentiment Analysis also known as Opinion Mining refers to the use of natural language processing, text analysis to systematically identify, extract, quantify, and study affective states and subjective information. Sentiment analysis is widely applied to reviews and survey responses, online and social media, and healthcare materials for applications that range from marketing to customer service to clinical medicine.    
  
In this project, we aim to perform Sentiment Analysis of product based reviews. Data used in this project are online product reviews collected from “amazon.com”. We expect to do review-level categorization of review data with promising outcomes.

Objective of the Project  
  
  Scrapping product reviews on various websites featuring various products specifically amazon.com.  
  Analyze and categorize review data.  
  Analyze sentiment on dataset from document level (review level).  
  Categorization or classification of opinion sentiment into-   Positive  
 Negative  
  

 System Design  
  
Hardware Requirements:  
•	Core i5/i7 processor  
•	At least 8 GB RAM  
•	At least 60 GB of Usable Hard Disk Space  
  
Software Requirements:  
•	Python 3.x  
•	Anaconda Distribution  
•	NLTK Toolkit  
•	UNIX/LINUX Operating System.  
  
Data Information:  
  
•	The Amazon reviews dataset consists of reviews from amazon. The data span a period of 18 years, including ~35 million reviews up to March 2013.  
Reviews include product and user information, ratings, and a plaintext review. For more information, please refer to the following paper: J. McAuley and J. Leskovec. Hidden factors and hidden topics: understanding rating dimensions with review text. RecSys, 2013.  
  
•	The Amazon reviews full score dataset is constructed by Xiang Zhang (xiang.zhang@nyu.edu) from the above dataset. It is used as a text classification benchmark in the following paper: Xiang Zhang, Junbo Zhao, Yann LeCun. Character-level Convolutional Networks for Text Classification. Advances in Neural Information Processing Systems 28 (NIPS 2015).  
  
•	The Amazon reviews full score dataset is constructed by randomly taking 200,000 samples for each review score from 1 to 5. In total there are 1,000,000 samples.  

Code:  
Loading the dataset:  
import json import pickle import numpy as np from matplotlib import pyplot as plt from textblob import TextBlob  
  
# fileHandler = open('datasets/reviews_digital_music.json', 'r')  
# reviewDatas = fileHandler.read().split('\n')  
# reviewText = []  
# reviewRating = []  
  
# for review in reviewDatas:  #   	if review == "":  
#   	  	continue  
#   	r = json.loads(review)  
#   	reviewText.append(r['reviewText']) 
#   	reviewRating.append(r['overall'])  
  
# fileHandler.close()  
# saveReviewText = open('review_text.pkl', 'wb')  
# saveReviewRating = open('review_rating.pkl','wb')  # pickle.dump(reviewText, saveReviewText) # pickle.dump(reviewRating, saveReviewRating) reviewTextFile = open('review_text.pkl', 'rb') reviewRatingFile = open('review_rating.pkl', 'rb') reviewText = pickle.load(reviewTextFile) reviewRating = pickle.load(reviewRatingFile)  
# print(len(reviewText))  
# print(reviewText[0])  
# print(reviewRating[0]) # ratings = np.array(reviewRating)   plt.hist(ratings, bins=np.arange(ratings.min(), ratings.max()+2)-0.5, rwidth=0.7) 
plt.xlabel('Rating', fontsize=14)  plt.ylabel('Frequency', fontsize=14)  plt.title('Histogram of Ratings', fontsize=18) plt.show() lang = {} i = 0 for review in reviewText:  
  	tb = TextBlob(review)   l = tb.detect_language()   if l != 'en':  
lang.setdefault(l, [])  
  	lang[l].append(i)   	 
 print(i, l)   	i += 1 print(lang)  
  
Scrapping data:  
from selenium import webdriver from 
selenium.webdriver.chrome.options import Options from bs4 import BeautifulSoup import openpyxl class Review():   	def __init__(self):  
  	self.rating=""   	self.info=""   	self.review=""  
def scrape():  
  options = Options()   options.add_argument("--headless") # Runs Chrome in headless mode.   options.add_argument('--no-sandbox') # # Bypass OS security model   options.add_argument('start-maximized')   options.add_argument('disableinfobars')   options.add_argument("--disable-extensions")   driver=webdriver.Chrome(executable_path=r'C:\chromedriver\chromedriver.exe')  
  	url='https://www.amazon.com/Moto-PLUS-5th-Generation-Exclusive/product- reviews/B0785NN142/ref=cm_cr_arp_d_paging_btm_2?ie=UTF8&reviewerType=all_reviews&pageNumb er=5'  
  	driver.get(url)  
  
  soup=BeautifulSoup(driver.page_source,'lxml')   ul=soup.find_all('div',class_='a-section review')   review_list=[]   for d in ul:  
  	  	a=d.find('div',class_='a-row')     sib=a.findNextSibling()  
  	  	b=d.find('div',class_='a-row a-spacing-medium review-data')  
  	  	'''print sib.text'''  
	new_r=Review()   	 
 new_r.rating=a.text   	  	new_r.info=sib.text    	new_r.review=b.text  
  	  	  
  	  	review_list.append(new_r)   driver.quit()   	return review_list def main():    
m = scrape() i=1 for r in m:  
  	  
  	  	book = openpyxl.load_workbook('Sample.xlsx')   	  	sheet = 
book.get_sheet_by_name('Sample Sheet')   	  	sheet.cell(row=i, column=1).value = r.rating    	sheet.cell(row=i, column=1).alignment = openpyxl.styles.Alignment(horizontal='center', vertical='center', wrap_text=True)  
    sheet.cell(row=i, column=3).value = r.info      sheet.cell(row=i, column=3).alignment = 
openpyxl.styles.Alignment(horizontal='center', vertical='center', wrap_text=True)   	  sheet.cell(row=i, column=5).value = r.review.encode('utf-8')   	  	sheet.cell(row=i, column=5).alignment = openpyxl.styles.Alignment(horizontal='center', vertical='center', wrap_text=True)  
  	  	book.save('Sample.xlsx')    	  	i=i+1     	 if 
__name__ == '__main__':      main()  
  
Preprocessing Data:  
import string from nltk.corpus import stopwords as sw from nltk.corpus import wordnet 
as wn from nltk import wordpunct_tokenize from nltk import sent_tokenize from nltk import WordNetLemmatizer from nltk import pos_tag class NltkPreprocessor:   def __init__(self, stopwords = None, punct = None, lower = True, strip = True):   	 
 self.lower = lower   	  	self.strip = strip  
self.stopwords = stopwords or set(sw.words('english'))  
  	self.punct = punct or set(string.punctuation)   	self.lemmatizer = WordNetLemmatizer()  
  	def tokenize(self, document):    	  	tokenized_doc = []  
  
  	  	for sent in sent_tokenize(document):   	  	  	for token, tag in pos_tag(wordpunct_tokenize(sent)):   	  	  	  	token = token.lower() if self.lower else token   	  	  	  	token = token.strip() if self.strip else token     	  	  	token = token.strip('_0123456789') if self.strip else token  
  	  	  	  	# token = re.sub(r'\d+', '', token)     	  	  	  	if token in self.stopwords:  
  	  	  	  	  	continue     	  	  	  	if all(char in self.punct for char in token):  
  	  	  	  	  	continue  
  
  	  	  	  	lemma = self.lemmatize(token, tag)   	  	  	  tokenized_doc.append(lemma)  
  
  	  	return tokenized_doc  
   	def lemmatize(self, token, tag):  
  	  	tag = {  
  	  	  	'N': wn.NOUN,  
  	  	  	'V': wn.VERB,  
  	  	  	'R': wn.ADV,  
  	  	  	'J': wn.ADJ  
  	  	}.get(tag[0], wn.NOUN)  
    return self.lemmatizer.lemmatize(token, tag)  Sentiment Analysis:  
 import ast import numpy as np import pandas as pd 
import re from nltk.corpus import stopwords from nltk.stem import SnowballStemmer from sklearn.model_selection import train_test_split  
from sklearn.feature_selection import SelectKBest, chi2, SelectPercentile, f_classif from 
sklearn.feature_extraction.text import TfidfVectorizer from sklearn.pipeline import Pipeline from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix from sklearn.svm import LinearSVC # from textblob import TextBlob from time import time  
  
def getInitialData(data_file):  
  	print('Fetching initial data...')    	t = time()  
  
  	i = 0   df = {}   	with open(data_file, 'r') as file_handler:    	  	for review in file_handler.readlines():    
  	df[i] = ast.literal_eval(review)  
  	  	  	i += 1  
  
  	reviews_df = pd.DataFrame.from_dict(df, orient = 'index')  
 reviews_df.to_pickle('reviews_digital_music.pickle')  print('Fetching data completed!')  print('Fetching time: ', round(time()-t, 3), 's\n')  
  
  
# def filterLanguage(text):  
#   	text_blob = TextBlob(text)  
#   	return text_blob.detect_language()  
  
def prepareData(reviews_df):  print('Preparing data...')  t = time()    	reviews_df.rename(columns = {"overall" : "reviewRating"}, inplace=True)  
 reviews_df.drop(columns = ['reviewerID', 'asin', 'reviewerName', 'helpful', 'summary', 'unixReviewTime', 'reviewTime'], inplace = True)  
  
  
  	reviews_df = reviews_df[reviews_df.reviewRating != 3.0] # Ignoring 3-star reviews -> neutral  
 reviews_df = reviews_df.assign(sentiment = np.where(reviews_df['reviewRating'] >= 4.0, 1, 0)) # 1 -> Positive, 0 -> Negative  
  
  	stemmer = SnowballStemmer('english')   stop_words = stopwords.words('english')  
  
  	# print(len(reviews_df.reviewText))  
  	# filterLanguage = lambda text: TextBlob(text).detect_language()  
  	# reviews_df = reviews_df[reviews_df['reviewText'].apply(filterLanguage) == 'en']   # print(len(reviews_df.reviewText))  
  
  	reviews_df = reviews_df.assign(cleaned = reviews_df['reviewText'].apply(lambda text: '  
'.join([stemmer.stem(w) for w in re.sub('[^a-z]+|(quot)+', ' ', text.lower()).split() if w not in stop_words])))   reviews_df.to_pickle('reviews_digital_music_preprocessed.pickle')  
  
  	print('Preparing data completed!')  
 print('Preparing time: ', round(time()-t, 3), 's\n')  
  
def preprocessData(reviews_df_preprocessed):  
 print('Preprocessing data...')  t = 
time()  
  	  
  	X = reviews_df_preprocessed.iloc[:, -1].values   y = reviews_df_preprocessed.iloc[:, -2].values  
  
  	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)  
  
  	print('Preprocessing data completed!')  
 print('Preprocessing time: ', round(time()-t, 3), 's\n')  
  
  	return X_train, X_test, y_train, y_test  
  
def evaluate(y_test, prediction):  
print('Evaluating results...')  
  	t = time()  
  	  
  	print('Accuracy: {}'.format(accuracy_score(y_test, prediction)))   print('Precision: {}'.format(precision_score(y_test, prediction)))   print('Recall: {}'.format(recall_score(y_test, prediction)))   	print('f1: {}'.format(f1_score(y_test, prediction)))  
  
  	print('Results evaluated!')  
 print('Evaluation time: ', round(time()-t, 3), 's\n')  
  
# getInitialData('datasets/reviews_digital_music.json')  
# reviews_df = pd.read_pickle('reviews_digital_music.pickle')  
  
# prepareData(reviews_df) reviews_df_preprocessed = pd.read_pickle('reviews_digital_music_preprocessed.pickle')  
# print(reviews_df_preprocessed.isnull().values.sum()) # Check for any null values  
  
X_train, X_test, y_train, y_test = preprocessData(reviews_df_preprocessed)  
  
print('Training data...') t 
= time()  
  
pipeline = Pipeline([  
  	  	  	  	('vect', TfidfVectorizer(ngram_range = (1,2), stop_words = 'english', sublinear_tf = True)),  
  	  	  	  	('chi', SelectKBest(score_func = chi2, k = 50000)),  
  	  	  	  	('clf', LinearSVC(C = 1.0, penalty = 'l1', max_iter = 3000, dual = False, class_weight = 'balanced'))  
  	  	  	])  
  
model = pipeline.fit(X_train, y_train)  
  
print('Training data completed!') print('Training 
time: ', round(time()-t, 3), 's\n')  
  
print('Predicting Test data...') t 
= time()  
  
prediction = model.predict(X_test)  
  
print('Prediction completed!')  print('Prediction time: ', round(time()-t, 3), 's\n')  
  
evaluate(y_test, prediction)  
  
print('Confusion matrix: {}'.format(confusion_matrix(y_test, prediction)))  print() l = (y_test == 0).sum() + (y_test == 
1).sum() s = y_test.sum()  print('Total number of observations: ' + str(l)) 
print('Positives in observation: ' + str(s)) print('Negatives in observation: ' + str(l - s))  print('Majority class is: ' + str(s / l * 100) + '%')  
  
Graph Plotting Code: import numpy as np import matplotlib.pyplot as plt from matplotlib.ticker import MaxNLocator from collections import namedtuple n_groups = 5  score_MNB = (85.25,  85.31, 85.56, 84.95, 85.31)  score_LR = (88.12,  88.05, 87.54, 88.72, 88.05)  score_LSVC=(88.12,  88.11, 87.59, 88.80, 88.11)  score_RF=(82.43,  81.82, 79.74, 85.30, 81.83)  
  
#n1=(score_MNB[0], score_LR[0], score_LSVC[0], score_RF[0])  
#n2=(score_MNB[1], score_LR[1], score_LSVC[1], score_RF[1])  
#n3=(score_MNB[2], score_LR[2], score_LSVC[2], score_RF[2])  
#n4=(score_MNB[3], score_LR[3], score_LSVC[3], score_RF[3])  #n5=(score_MNB[4], score_LR[4], score_LSVC[4], score_RF[4]) fig, ax = plt.subplots() index = np.arange(n_groups) bar_width = 0.1 opacity = 0.7 error_config = {'ecolor': '0.3'} rects1 = ax.bar(index,score_MNB, bar_width,                 alpha=opacity, color='b',                  error_kw=error_config,                 label='Multinomial Naive Bayes') z=index + bar_width rects2 = ax.bar(z, score_LR, bar_width,                 alpha=opacity, color='r',                 error_kw=error_config,                 label='Logistic Regression') z=z+ bar_width  
rects3 = ax.bar(z, score_LSVC, bar_width,                 
alpha=opacity, color='y',                 error_kw=error_config,                 label='Linear SVM') z=z+ bar_width  
rects4 = ax.bar(z, score_RF, bar_width,                 
alpha=opacity, color='g',                 error_kw=error_config,                 label='Random Forest') ax.set_xlabel('Score Parameters') ax.set_ylabel('Scores (in %)') ax.set_title('Scores of Classifiers') ax.set_xticks(index + bar_width / 2)  
ax.set_xticklabels(('F1', 'Accuracy', 'Precision', 'Recall', 'ROC AUC')) ax.legend(bbox_to_anchor=(1, 1.02), loc=5, borderaxespad=0) fig.tight_layout() plt.show()  
  

