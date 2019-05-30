#!/usr/bin/env python
# coding: utf-8


## this script is for words correction
## input of thr data is already abbreviation extended
## words are identified as misspell if more than one word has small levistein distance to others, or same stems
## corrected form of words are denoted as the word with max. frequency within same Levenshtein distance group
## output is a csv format of the data
## corrected and wrongly spelled words are calculated and stored in seperate files


from io import StringIO
import pandas as pd
import string
from iteration_utilities import deepflatten
import nltk




inp_path = '/validation_extended.csv'  
corrected_words_dat_path = '/df_allvisits_correction.csv'   ## file storing corrected and wrongly spelled words
out_path = '/validation_corrected.csv' 

left_eye = 'full_left'   ## input smart text column for correction (abreviation extended)
right_eye = 'full_right'  ## input smart text column for correction (abreviation extended)

left_eye_cor = 'corrected_left'    ## output smart text column after spelling correction: left eyes
right_eye_cor = 'corrected_right'    ## output smart text column after spelling correction: right eyes




dat=open(inp_path,errors='ignore')
dat=dat.read()
test = StringIO(dat)
df = pd.read_csv(test, sep=",",na_values=" ")




correction = pd.read_csv(corrected_words_dat_path, sep=",",na_values=" ")


## list of corrected words: data from wide to long
correction=correction[correction['sum(cts)']>1][['corrected','word']]

correction=correction[correction.word.apply(lambda word: word.count(',')>=1)]

wd_lst=list(map(lambda i:[correction.loc[i].corrected,
                       correction.loc[i].word.strip().split(', ')],correction.index))

long_lst=[]
for x in wd_lst:
    long_lst.append(list(map(lambda i:[x[0],x[1][i]],range(0,len(x[1])))))

long_lst=list(deepflatten(long_lst, depth=1))
pd_words=pd.DataFrame(long_lst,columns=['correct_spell','wrong_spell'])



def tokenizer(sent):
    '''tokenization, split puntuaction from tokens'''
    tokens = nltk.word_tokenize(sent)
    tokens = list(map(lambda x: x.strip(string.punctuation),tokens))
    return tokens

def correction(sentence, 
               dat=pd_words):
    '''
        replace wrong spell with correct spell
        future work: if common==0, print out [code for now didn't consider where words are wrongly spelled but not in wrong_sp]
    '''
    tokens=tokenizer(sentence)
    wrong_sp=dat.wrong_spell.values.tolist()
    common=list(set(tokens) & set(wrong_sp))
    if len(common)!=0:
        for wd in common:
            sentence=sentence.replace(wd, dat[dat['wrong_spell']==wd].correct_spell.values[0])
    return sentence

df[right_eye] = df[right_eye].astype(str)
df[left_eye] = df[left_eye].astype(str)

corrected_R_descr=list(map(lambda x: correction(x),df[right_eye].values.tolist()))
corrected_L_descr=list(map(lambda x: correction(x),df[left_eye].values.tolist()))

df[right_eye_cor]=pd.Series(corrected_R_descr,index=df.index)
df[left_eye_cor]=pd.Series(corrected_L_descr,index=df.index)




df.to_csv(out_path)






