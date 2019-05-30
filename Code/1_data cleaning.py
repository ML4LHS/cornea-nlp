#!/usr/bin/env python
# coding: utf-8

#The Algorithm for data cleaning:
# This script is for data cleaning:
# 1. combining descriptions(free text) for encounters with same day visits and same encounter
# 2. drop duplicate encounters at same visiting time for the same patient
# 3. drop encounters with no description on both eyes
#
# Input: encounter_notes_12.csv
#
# Output: cf_dat.csv

import pandas as pd
from io import StringIO
import numpy as np
import string
import re


#inp_path = '/Users/huantan/google drive/kellogg/qualitative analysis/inp_data/Random selected NEW encounters VALIDATION (3.28.2019).csv' 
inp_path='New encounters validation.csv'
## your_csv_file.csv
#out_path = '/Users/huantan/google drive/kellogg/qualitative analysis/inp_data/validation_cleaned_05.28.2019.csv' 
out_path='validated_cleaned.csv'
## your_csv_output_file.csv
left_eye = 'SLE_L_CORNEA_1020'
right_eye = 'SLE_R_CORNEA_1013'
patid = 'PAT_ID'
visit_time = 'CUR_VALUE_DATETIME'
encounter_identifier = 'DX_SOURCE_ID'
dat=open(inp_path,errors='ignore')
dat=dat.read()
test = StringIO(dat)
df = pd.read_csv(test, sep=",",na_values=" ")


for i in df.columns:
    df[[i]] = df[[i]].astype('str')


colname = list(df.columns)
colname.remove(right_eye)
colname.remove(left_eye)

## replace missing values as empty string
df=df.replace('nan','')

## join encounters for the same visiting (same encounter information besides text)
df1=df.groupby(colname,as_index=False).agg(lambda col: ''.join(col))


## drop duplicated encounters with same encounter identifier and visit time
df=df[df.duplicated(subset=[encounter_identifier,visit_time],keep='first')==False]

## store visiting time information
df['time']=pd.Series(list(map(lambda x: int(x.split()[1].replace(':','')),df[visit_time])),index=df.index)

## aggregate text data with same encounter identifier, but different times in a day
## the max visiting time is taken to be new visiting time for aggregated data
colname.remove(visit_time)
datetime=df.sort_values(['time'],ascending=False)[df[encounter_identifier].isin(df[df.duplicated(subset=[encounter_identifier])][encounter_identifier].values.tolist())].groupby(colname,as_index=False)[visit_time]\
.max()

description=df.sort_values(['time'],ascending=False)[df[encounter_identifier].isin(df[df.duplicated(subset=[encounter_identifier])][encounter_identifier].values.tolist())].groupby(colname,as_index=False)\
.agg(lambda col: ''.join(col))

description=description.drop(labels=visit_time,axis=1)

df=df.drop(labels=['time'],axis=1)

description[visit_time]=pd.Series(datetime[visit_time],index=description.index)

## storing information back to data
df=pd.concat([df[df[encounter_identifier].isin(description[encounter_identifier])==False],description],ignore_index=True)


## remove encounter with no text information stored in both eyes
#df=df.drop(df[df[right_eye].apply(
#        lambda r:r.lower().replace(" ","")=='')][df[left_eye].apply(
#        lambda l:l.lower().replace(" ","")=='')].index)

df[right_eye]=df[right_eye].str.lower()
df[left_eye]=df[left_eye].str.lower()

## if one of left eye or right eye free text has no information, that means it's clear
for i in df.index:
    for j in [right_eye, left_eye]:
        if df.loc[i,j].lower().replace(" ","")=="" or df.loc[i, j].lower().replace(" ","")=="nan":
            df.loc[i,j]='clear'

## clean cases where mm is connected to other words or digits
## input should be string of sentences
def sep_mm(sent):
    words=sent.strip().split()   
    for i in words:
        if len(i)>2 and i[:2]=='mm':
            sent=sent.replace(i, 'mm'+' '+str(i[2:]))
        elif len(i)>2 and i[-2:]=='mm':
            sent=sent.replace(i, str(i[:-2]) + ' ' + 'mm')
        elif len(i)>=3 and (i[-2:]=='mmv' or i[-2:]=='mmh'):
            sent=sent.replace(i, str(i[:-2]) + 'mm '+ i[-1])
    return sent

## unify clock pattern
def sep_oclock(sent):
    if 'o clock' in sent:
        sent=sent.replace('o clock','oclock')
    elif "o'clock" in sent:
        sent=sent.replace("o'clock",'oclock')
    return sent


##spliting times and floats
df[right_eye]=pd.Series(
    list(map(lambda x: re.sub( r'(x)(\d*\.?\d+)', r'\1 \2', x),
             df[right_eye].values)),index=df.index)
df[left_eye]=pd.Series(
    list(map(lambda x: re.sub( r'(x)(\d*\.?\d+)', r'\1 \2', x ),
             df[left_eye].values)),index=df.index)

##spliting times and floats
df[right_eye]=pd.Series(
    list(map(lambda x: re.sub( r'(\d*\.?\d+)(x)', r'\1 \2', x ),
             df[right_eye].values)),index=df.index)
df[left_eye]=pd.Series(
    list(map(lambda x: re.sub( r'(\d*\.?\d+)(x)', r'\1 \2', x ),
             df[left_eye].values)),index=df.index)

extended=list(map(lambda x: sep_mm(x),df[right_eye]))
df[right_eye]=pd.Series(extended,index=df.index)
extended=list(map(lambda x: sep_mm(x),df[left_eye]))
df[left_eye]=pd.Series(extended,index=df.index)

extended=list(map(lambda x: sep_oclock(x),df[right_eye]))
df[right_eye]=pd.Series(extended,index=df.index)
extended=list(map(lambda x: sep_oclock(x),df[left_eye]))
df[left_eye]=pd.Series(extended,index=df.index)




df.to_csv(out_path)

