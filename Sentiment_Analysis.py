import pandas as pd
import re
import string
import nltk
import itertools
import tweepy
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn  import metrics
from sklearn.metrics import classification_report
from textblob import TextBlob
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB

###############################################################################

consumerKey="ZJMKiHDs5KjXN251qvde0ddY4"
consumerSecret="5auNnTfhjeTyxYp0wI0KeFwcoum5f2IfNcrxTzUGy3AOJ9d8PC"
accessToken="991949084-2IAqszBXBsDiVHE2NFx61yIrXBQHDiZBW4HewxaZ"
accessTokenSecret="uQFLRvkdBWsFuquQ9sTuptEupr9r71SL55d1RH38DVG8L"
            
auth=tweepy.OAuthHandler(consumer_key=consumerKey, consumer_secret=consumerSecret)
auth.set_access_token(accessToken,accessTokenSecret)
api=tweepy.API(auth)

searchTerm=input("enter keyward/hashtag to search about:")
noOfSearchTerms=int(input("enter how many tweets to analyze:"))

tweets=tweepy.Cursor(api.search, q=searchTerm).items(noOfSearchTerms)

s={'sentiment':[]}
dfS = pd.DataFrame(s)

d={'tweet':[]}
df = pd.DataFrame(d)
cp=0
cneu=0
cn=0
for tweet in tweets:

    sentiment = TextBlob(tweet.text)

    if sentiment.sentiment.polarity < 0:
        p='negative'
        cn +=1;
    elif sentiment.sentiment.polarity == 0:
        p='neutral'
        cneu +=1;
    else:
        p='positive'
        cp +=1;
        
    dfS=dfS.append({'sentiment':p},ignore_index= True)
    
    m = tweet.text 
   
    df=df.append({'tweet':m},ignore_index= True)
    

string.punctuation #it will find all the punctuation in a string ex !@#$%^&*..
###############################################################################

#REMOVE MENTIONS,URLS AND NONALPHANUMERIC TEXT

def remove_mentionsUrls(text): #removes mentions and urls from string
    tweet_out=re.sub(r'@[A-Za-z0-9]+','',text)
    re.sub('https?://[A-Za-z0-0./]+','',tweet_out)
    return tweet_out

def remove_nonalphanumeric(text):
    text_out="".join([char for char in text if char not in string.punctuation])
    return text_out

df['Tweet_noment_nourl']=df['tweet'].apply(lambda x:remove_mentionsUrls(x))

df['Tweet_nopunc']=df['Tweet_noment_nourl'].apply(lambda x:remove_nonalphanumeric(x))

print ("ORIGINAL TWEETS")

print ("REMOVE URLS AND MENTIONS")
print(df.head())
###############################################################################

#TOKENIZING

print ("TOKENIZING")
def tokanization(text): #break into words and make them all into lowercase
    tokens=re.split('\W+',text)
    return tokens


df['Tweet_tokens']=df['Tweet_nopunc'].apply(lambda x:tokanization(x.lower()))

d=df.head()
print(d)
###############################################################################

#STEMMING

ps=nltk.PorterStemmer()

print ("STEMMING")
def stemming(text):
    out_text=[ps.stem(word) for word in text]
    return out_text
df['FirstDataSetStem']=df['Tweet_tokens'].apply(lambda x: stemming(x))

print(df['FirstDataSetStem'].head())
###############################################################################

#STOPWORDS

nltk.download('stopwords') #imported from nltk library
stopword=nltk.corpus.stopwords.words('english')

def remove_stopwords(tokanized_list):
    text_out=[word for word in tokanized_list if word not in stopword]
    return text_out

df['SecondDataSet']=df['FirstDataSetStem'].apply(lambda x: remove_stopwords(x))
print(df['SecondDataSet'].head)
###############################################################################

#Removing words with fewer frequencies
print('FEWER FREQUENCIES')

flat_list=list(itertools.chain.from_iterable(df['SecondDataSet'])) #puts it all into one lilst
fd=nltk.FreqDist(flat_list)
word_toKeep=list(filter(lambda x: 2000>x[1]>3,fd.items())) #filter words >5 and <2000 frequencies
word_list_ToKeep=[item[0] for item in word_toKeep] #words that we keep
def remove_lessfreq(tokanized_tweets):
    text_out=[word for word in tokanized_tweets if word in word_list_ToKeep]
    return text_out
df['ThirdDataSet']=df['SecondDataSet'].apply(lambda x: remove_lessfreq(x))
ListofUniqueWords=set(list(itertools.chain.from_iterable(df['ThirdDataSet'])))
print(df['ThirdDataSet'].head)
print(len(ListofUniqueWords))
###############################################################################
#Create list from data set

def join_tokens(tokens):
    document=" ".join([word for word in tokens if not word.isdigit()])
    return document

df['ThirdDataSet_Documents']=df['ThirdDataSet'].apply(lambda x:join_tokens(x))
print(df['ThirdDataSet_Documents'].head())
#print(ListofUniqueWords)
###############################################################################

cv=CountVectorizer(ListofUniqueWords)
countvector=cv.fit_transform(df['ThirdDataSet_Documents'])
#print(countvector.shape)

countvectorDF=pd.DataFrame(countvector.toarray())
countvectorDF.columns=cv.get_feature_names()

print(countvectorDF[:10])

print(dfS)
###############################################################################
x= countvectorDF
y= dfS
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)
print(X_train)
##############################################################################
#naive baye model
naive_baye = MultinomialNB()
naive_baye.fit(X_train,y_train)
pred = naive_baye.predict(X_test)
##############################################################################
#svm model
svm = svm.SVC(kernel = 'linear')
svmm = svm.fit(X_train,y_train)
ypredict = svm.predict(X_test)
 
###############################################################################
logreg = LogisticRegression()
loreg = logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)
###############################################################################
#check Accuracy
cnf_matrix = metrics.confusion_matrix(y_test,y_pred)
cnf_matrixSVM = metrics.confusion_matrix(y_test,ypredict)
cnf_matrixNB = metrics.confusion_matrix(y_test,pred)

###############################################################################
print(classification_report(y_test,y_pred))
print("Accuracy Logestic model:",metrics.accuracy_score(y_test,y_pred))

print(classification_report(y_test,ypredict))
print("Accuracy of SVM:",metrics.accuracy_score(y_test,ypredict))

print(classification_report(y_test,ypredict))
print("Accuracy of NB:",metrics.accuracy_score(y_test,pred))
##############################################################################
#plot bar graph
label = ['Positive', 'Neutral','Negative']
no_polarity = [cp,cneu,cn]
plt.bar(label,no_polarity, color = ['green','gray','red'])
plt.xlabel('Polarity')
plt.ylabel('Number of Polarities')
plt.title('Number of tweets based on sentiment')
plt.show


