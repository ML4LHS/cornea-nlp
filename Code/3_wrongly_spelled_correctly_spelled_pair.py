#!/usr/bin/env python
# coding: utf-8




import pandas as pd
import numpy as np
import string
from io import StringIO
import time
from functools import reduce
import itertools
import operator
import pyspark
from pyspark.sql.types import *
from pyspark.sql import *
from pyspark import SparkConf, SparkContext
from pyspark.sql.functions import col
import nltk
from nltk import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords as nltkstopwords
from nltk.corpus import wordnet
from nltk.stem.porter import *
from nltk import edit_distance
from autocorrect import spell

get_ipython().run_line_magic('matplotlib', 'inline')
nltk.download('stopwords')
nltk.download('wordnet')



##data readin
inp_path = 'encounter_notes_12.csv'
dat=open(inp_path,errors='ignore',)
dat=dat.read()
test = StringIO(dat)
df = pd.read_csv(test, sep=",",na_values=" ")



left_eye ='SLE_L_CORNEA_1020'
right_eye ='SLE_R_CORNEA_1013'
patid = 'PAT_ID'

## combine left and right eye description
df['description'] = df[left_eye].map(str) +' '+ df[right_eye].map(str)


## setting spark environment
conf = SparkConf().setAppName("wd_count")
sc = SparkContext(conf=conf)
sqlCtx = SQLContext(sc)
ps_df=sqlCtx.createDataFrame(df[[patid,'description']])

## from pyspark.df to pyspark rdd, get word frequency
Rdd=ps_df.rdd
wd_ct=Rdd.map(lambda x: [x[0],x[1].lower().strip().split()]).flatMap(lambda x: [tuple([x[0], x[1][i].strip(string.punctuation)]) for i in range(0,len(x[1]))]).map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y).sortBy(lambda x: -x[0]).map(lambda x:[x[0][0],x[0][1],x[1]]).toDF()


## print col name
wd_ct.printSchema()

## rename cols
wd_ct = wd_ct.selectExpr("_1 as PAT_ID","_2 as word", "_3 as cts")

## aggregate words together by summing frequency
words=wd_ct.groupBy("word").agg({"cts": "sum"}).sort(col("sum(cts)").desc())


## transform to pandas df
pd_words=words.toPandas().sort_values('word')
pd_words.sort_values('word').head(10)


#correction('wiht')
#start = timeit.timeit()
#newlist = list(map(correction, pd_words['word'].tolist()))
#end = timeit.timeit()
#print(end - start)




## tokenizing
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
tokens = tokenizer.tokenize(' '.join(pd_words['word'].tolist()))
pd_words=pd_words.loc[pd_words['word'].isin(tokens)]

##spelling correction
start = time.time()
corrected=list(map(spell, pd_words['word'].tolist()))
end = time.time()
print(end-start)
pd_words['corrected']=pd.Series(corrected,index=pd_words.index)

##remove stopwords
nonstopwords = [wd for wd in corrected if wd not in nltkstopwords.words('english')]
pd_words=pd_words.loc[pd_words['corrected'].isin(nonstopwords)]

##stemming
stemmer = PorterStemmer()
words1 = [stemmer.stem(word) for word in pd_words['corrected'].tolist()]
pd_words['stemmer']=pd.Series(words1,index=pd_words.index)

#lmtzr = WordNetLemmatizer()
#words2 = [lmtzr.lemmatize(word) for word in pd_words['corrected'].tolist()]
#pd_words['lmtzr']=pd.Series(words2,index=pd_words.index)


#nonstopwords = [wd for wd in pd_words.word if wd not in nltkstopwords.words('english')]
#pd_words=pd_words.loc[pd_words['corrected'].isin(nonstopwords)]


## aggregate words with same stemmer
a=pd_words.groupby('stemmer')['word'].apply(lambda x: ', '.join(x)).to_frame()
b=pd_words.groupby('stemmer')['sum(cts)'].sum().to_frame()
combined= pd.concat([a, b], axis=1)


combined=combined[combined.index.isin(['nan'])==False ]
combined=combined.reset_index()


def Prob(word, N=sum(pd_words['sum(cts)'])):
    "Probability of `word`."
    return pd_words[pd_words.word==word]['sum(cts)'].values/ N

def correction(lst_of_word): 
    "Most probable spelling correction for word."
    return max(lst_of_word, key=Prob)

corrected=[]
for i in range (0, len(combined)):
    corrected.append(correction(combined.word.iloc[i].split(', ')))




combined['stemmed_corrected']=pd.Series(corrected, index=combined.index)
cols=['stemmer','stemmed_corrected','word', 'sum(cts)']
combined = combined[cols]


newlist=combined.stemmer.tolist()
needed=list()
for i in newlist:
    if len(i)>=2:
        needed.append(i)

combined=combined[combined.stemmer.isin(needed)==True ]

def closed_wd(lst):
    '''find close words using leveinshtein distance'''
    pairs = [[lst[w1], lst[w2]] for w1 in range(len(lst)) for w2 in range(w1+1,len(lst))]
    closed_pairs=list()
    for i in pairs:
        if edit_distance(i[0], i[1])<max(len(i[0])/5,len(i[1])/5):
            if i[0][:2]==i[1][:2]:
                i.sort()
                closed_pairs.append(i)
    closed_pairs = [list(x) for x in set(tuple(x) for x in closed_pairs)]
    LL = set(itertools.chain.from_iterable(closed_pairs)) 
    for each in LL:
        components = [x for x in closed_pairs if each in x]
        for i in components:
            closed_pairs.remove(i)
            closed_pairs += [list(set(itertools.chain.from_iterable(components)))]
    closed_pairs = [list(x) for x in set(tuple(x) for x in closed_pairs)]
    return closed_pairs
#closed_wd(combined.stemmed_corrected.tolist())


newlist=combined.stemmed_corrected.tolist()
sim=closed_wd(newlist)


#newlist=combined.stemmer.tolist()
#simil_list3=closed_wd(newlist)


sub=combined[combined.stemmed_corrected.isin(reduce(operator.concat, sim))]
combined=combined[combined.stemmed_corrected.isin(reduce(operator.concat, sim))==False]



## assign same group to similar words

groups=list(['na']*sub.shape[0])
for j in range(0,len(sub)):
    for i in range(0,len(sim)):
        if sub.stemmed_corrected.iloc[j] in sim[i]:
            groups[j]=i


sub['groups'] = pd.Series(groups, index=sub.index)



## aggregation
a=sub.groupby('groups')['stemmer'].apply(lambda x: ', '.join(x)).to_frame()
b=sub.groupby('groups')['word'].apply(lambda x: ', '.join(x)).to_frame()
c=sub.groupby('groups')['sum(cts)'].sum().to_frame()
d=sub.groupby('groups')['stemmed_corrected'].apply(lambda x: ', '.join(x)).to_frame()

grouped_sub= pd.concat([a, b,c,d], axis=1)


## updating corrected word by frequency
corrected=[]
for i in range (0, len(grouped_sub)):
    corrected.append(correction(grouped_sub.word.iloc[i].split(', ')))
grouped_sub['stemmed_corrected']= pd.Series(corrected, index=grouped_sub.index)
grouped_sub['stemmer']=pd.Series([stemmer.stem(word) for word in corrected], index=grouped_sub.index)

combined=combined.append(grouped_sub, ignore_index=True)


combined=combined[['stemmer','stemmed_corrected','word','sum(cts)']].sort_values('sum(cts)',ascending=False)


epi_index=combined[combined.stemmed_corrected=='epi'].index.values[0]
combined.loc[epi_index,'sum(cts)']

defect_index=combined[combined.stemmed_corrected=='defect'].index.values[0]
combined.loc[defect_index,'sum(cts)']


if 'epidefect' in combined.stemmed_corrected.tolist():
    cts=combined[combined.stemmed_corrected=='epidefect']['sum(cts)'].values[0]
    epi_index=combined[combined.stemmed_corrected=='epi'].index.values[0]
    defect_index=combined[combined.stemmed_corrected=='defect'].index.values[0]
    combined = combined[combined.stemmed_corrected != 'epidefect']
    combined.loc[epi_index,'sum(cts)']=combined.loc[epi_index,'sum(cts)']+cts
    combined.loc[defect_index,'sum(cts)']=combined.loc[defect_index,'sum(cts)']+cts
    




combined.to_csv('df_allvisits_correction.csv', sep=',')



