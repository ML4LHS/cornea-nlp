#!/usr/bin/env python
# coding: utf-8


## this script is for extending abreviations unique for data inputted (opthomology words extention)
## abbreviated words are preidentified by finding len(word) <=5 and count(word) >1 in the text data, and then manually went through by cornea specialists
## possible abbreviation and full name pairs are stored at "abbreviation_extension.csv"



from io import StringIO
import pandas as pd
import string
import re
import timeit



dat_abbreviation='/abbreviation extension.csv'  ## data storing abbreviation and full name pairs
inp_path = '/validation_cleaned.csv'  ##input data, smart text before abbreviation extension
out_path = '/validation_extended.csv'  ##output data, smart text after abbreviation extension

left_eye = 'SLE_L_CORNEA_1020'  ## input smart text column for abbreviation extension, left eyes
right_eye = 'SLE_R_CORNEA_1013'  ## input smart text column for abbreviation extension, right eyes

left_eye_extended = 'full_left'    ##  smart text column output after abbreviation extension: left eyes
right_eye_extended = 'full_right'    ## smart text column output after abbreviation extension: right eyes






dat=open(inp_path,errors='ignore',)
dat=dat.read()
test = StringIO(dat)
df = pd.read_csv(test, sep=",",na_values=" ")

dat=open(dat_abbreviation,errors='ignore',)
dat=dat.read()
test = StringIO(dat)
abbreviation = pd.read_csv(test, sep=",",na_values=" ")




def tokenizer(sent):
    '''tokenization by any punctuation'''
    ## .*?\S.*? is a pattern matching anything that is not a space
    ## $ is added to match last token in a string if it's a punctuation symbol.
    tokens = [t.strip() for t in re.findall(r'\b.*?\S.*?(?:\b|$)', sent)]
    return tokens

def abbrev_exten(sent,dat2=abbreviation):
    '''abbreviation extension for a single sentence'''
    ## replace abbreviation with full version if any, else return original sentence
    new_sent=[]
    tokens=tokenizer(sent)
    short=dat2['abbrev.'].values.tolist() 
    
    common=list(set(tokens) & set(short))  ## check any common tokens between sentence and abbreviation
    if len(common)!=0:
        for i in tokens:
            if i in common:
                new_sent.append(dat2[dat2['abbrev.']==i].Extended.values[0])
            else:
                new_sent.append(i)
        new_sent=' '.join(new_sent)
    else:
        new_sent=sent
    return new_sent
        




df[left_eye] = df[left_eye].astype(str).str.lower()
extended=list(map(lambda x: abbrev_exten(x),df[left_eye]))
df[left_eye_extended]=pd.Series(extended,index=df.index)

df[right_eye] = df[right_eye].astype(str).str.lower()
extended=list(map(lambda x: abbrev_exten(x),df[right_eye]))
df[right_eye_extended]=pd.Series(extended,index=df.index)


df[right_eye_extended] = pd.Series(list(map(lambda x: re.sub(r'(\d+) ([.:-]) (\d+)', r'\1\2\3', x),
  df[right_eye_extended].values)),index=df.index)
df[right_eye_extended] = pd.Series(list(map(lambda x: re.sub(r' ([%+,.]) ', r'\1 ', x),
  df[right_eye_extended].values)),index=df.index)
df[right_eye_extended] = pd.Series(list(map(lambda x: x.replace(' / ', '/').replace('  ',' ').replace('  ', ' '),
  df[right_eye_extended].values)),index=df.index)


df[left_eye_extended] = pd.Series(
    list(map(lambda x: re.sub( r'(\d+) ([.:-]) (\d+)', r'\1\2\3', x),
             df[left_eye_extended].values)),index=df.index)

df[left_eye_extended] = pd.Series(
    list(map(lambda x: re.sub(r' ([%+,.]) ', r'\1 ', x),
             df[left_eye_extended].values)),index=df.index)
df[left_eye_extended] = pd.Series(
    list(map(lambda x: x.replace(' / ', '/').replace('  ',' ').replace('  ', ' '),
             df[left_eye_extended].values)),index=df.index)




df.to_csv(out_path)






