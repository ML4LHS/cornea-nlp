#!/usr/bin/env python
# coding: utf-8

# #### functions in this script are created searching for keywords related measures

import pandas as pd
from nltk import tokenize
from iteration_utilities import deepflatten
import re
import statistics as stat

 




def hasNumbers(inputString):
    '''check if string has numbers'''
    return any(char.isdigit() for char in inputString)




def split_para(paragraph):
    '''
        Input: paragraph string
        Usage: splitting paragraph into segmentations, splitters are ",;" "with", "and"
        Output: list of spliited sentences
    '''
    try:
        sents = tokenize.sent_tokenize(paragraph)
    except TypeError: 
        print('expecting a string input')
    sents_new = list(map(lambda x:x.split('with '), sents))
    sents_new = list(deepflatten(sents_new,types=list))  ##flatten list of lists
    sents_new = list(map(lambda x:x.split(' and '), sents_new))
    sents_new = list(deepflatten(sents_new,types=list))
    sents_new = list(map(lambda x:x.split('surround'), sents_new))
    sents_new = list(deepflatten(sents_new,types=list))
    sents_new = list(map(lambda x:x.split(' but '), sents_new))
    sents_new = list(deepflatten(sents_new,types=list))
    
    return sents_new
    




def size_determ(rgx,keyword,sent):
    '''
        Input:
            rgx: list of regex patterns
            keyword: keyword to look for
            sent: sentence
        Output: return 1 if any rgx pattern along with keyword appears in sentence, else 0
    '''
    if rgx!=[]:
        for i in rgx:
            if i.findall(sent)!=[]:
                return 1
    if keyword in sent and hasNumbers(sent)==True:
        return 1
    else:
        return 0    ## no size determination
        




def apply_f(a,f):
    '''
        Input:
            a: list of lists
            f: function
        Usage: apply function to each string element in list of lists
    '''
    if isinstance(a,list):
        return list(map(lambda t:apply_f(t,f), a))
    else:
        return f(a)




def rm_post_culture(sent):
    '''
        Input:
            sent: sentence
        Usage: exclude segments about post culture/after culture
    '''
    if "post culture" in sent or "after culture" in sent:
        return ""
    else:
        return sent




def match_numbers(lst1,match_or_not):
    '''
        Input:
            lst1: list of measurements
            mactch_or_not: binary indicators from function size_determ(); 0 means no related measurement, 1 otherwise
        Output: list of valid measurements related to keywords
    '''
    if match_or_not!=0:
        return lst1
    elif lst1==['0','0']:
        return lst1
    else:
        return []




def no_measures(rgx_lst, sent_lst,measure_lst):
    '''
        Input:
            rgx_lst: list of regex patterns to look for (ex: no infiltrate)
            sent_lst: list of splitted segments
            measure_lst: list of measurements found within sent_lst
        Output: measure_lst as 0 if any regex pattern matched, else stay the same
    '''
    for j in rgx_lst:
        for i in range(0,len(sent_lst)):
            if j.findall(sent_lst[i])!=[]:
                measure_lst[i]=['0','0']
    return measure_lst





def rm_specialfmt(lst):
    '''
        Input:
            lst: list of numerical tokens matched
        Usage: deleting unwanted numerical measures such as time, data, percentage and so on
        Output: list of validated numerical measures
    '''
    re_time = re.compile(r'\d+\:\d+')
    re_date = re.compile(r'\d+\/\d+\/\d+')
    re_time1 = re.compile(r'\d+\-\d+')
    re_pct = re.compile(r'\d+%')
    re_number = re.compile(r'\d{3,}')
    re_plus = re.compile(r'\d+\+')
    
    if len(lst)==0:
        return lst
    else:
        new = [x for x in lst if not re_time.search(x) \
               and not re_date.search(x) and not re_pct.search(x) \
               and not re_number.search(x) and not re_time1.search(x) and not re_plus.search(x)]
        
        return new




def assign_values(dat,col_inp,col1,col2):
    '''
        Input:
            dat: dataframe
            col_inp: text column
            col1: column name for left eye measurement
            col2: column name for right eye measurement
        Usage: aggregate above functions, and assign values to dataframe column
        Output: a panda dataframe
    '''
    dat[col1] = pd.Series(['na']*len(dat),index=dat.index)
    dat[col2] = pd.Series(['na']*len(dat),index=dat.index)
    for i in dat.index:
        if dat.loc[i,col_inp]==[]:
            dat.loc[i,col1]='na'
            dat.loc[i,col2]='na'       
        elif len(dat.loc[i,col_inp])==1:
            new = rm_specialfmt(dat.loc[i,col_inp][0])
            if len(new) >= 2:
                dat.loc[i,col1]=new[0]
                dat.loc[i,col2]=new[1]
            elif len(new) == 1:
                dat.loc[i,col1]=new[0]
                dat.loc[i,col2]=new[0]   
            elif len(new) == 0:
                dat.loc[i,col1]='na'
                dat.loc[i,col2]='na'
        elif dat.loc[i,col_inp]!='na' and len(dat.loc[i,col_inp]) >= 2:
            measures = []
            for lsts in range(0,len(dat.loc[i,col_inp])):
                measures.append(rm_specialfmt(dat.loc[i,col_inp][lsts]))
            measures=apply_f(measures,eval)
            measures = list(filter(None, measures))
            if measures == []:
                dat.loc[i,col1]='na'
                dat.loc[i,col2]='na'
            else:
                mean = list(map(stat.mean,measures))
                idx = mean.index(max(mean))
                if len(measures[idx])>=2:
                    dat.loc[i,col1]=measures[idx][0]
                    dat.loc[i,col2]=measures[idx][1]
                elif len(measures[idx])==1:
                    dat.loc[i,col1]=measures[idx][0]
                    dat.loc[i,col2]=measures[idx][0]
    return dat




def reduce(dat, inp_col):
    '''
        input:
            dat: dataframe
            inp_col: text column
        usage: take the maximum measure if there are more than one related measures
        output: a panda dataframe
    '''
    for i in dat[dat[inp_col].apply(lambda x:len(x)>=2)].index.tolist():   
        measures=apply_f(dat.loc[i,inp_col],eval)
        measures = list(filter(None, measures))
        if measures == []:
            dat.loc[i,inp_col]='na'
        else:
            mean = list(map(stat.mean,measures))
            idx = mean.index(max(mean))
            dat.loc[i,inp_col] = str([measures[idx]])
    return dat



## commonly appeared 0 cases in the training set, ex: no defect, healed defect

key_wd = "defect"
ed_rgx = [re.compile(r'no (\w* ){0,}defect|without (\w* ){0,}defect'),
       re.compile(r'epi\w+ ((?!not)\w*(%)? ){0,}intact|intact (\w* ){0,}epi\w+'),
      re.compile(r'epi\w+ (defect)? ((?!nearly)(?!almost)\w* ){0,}healed|healed (epi\w+)? defect'),
      re.compile(r'epithelial defect ((?!almost)(?!mostly)\w* ){0,}resolved|resolved (\w* ){0,}(epithelial )?defect'),
      re.compile(r'(no|without|resolved|negative) (\w* ){0,}stain(ing)?'),
      re.compile(r'epi\w+ irregularity'), re.compile(r'pinpoint (\w* ){0,}epithelial defect'),
       re.compile(r'(\w*(?!no) ){0,}epithelial erosion'),re.compile(r'pinpoint (\w*(?!no) ){0,}stain'),
       re.compile(r'punctate (\w* ){0,}epi\w+ defect'),re.compile(r'epithelium healed'),
      re.compile(r'punctate stain'),re.compile(r'defect (\w*(?!no) ){0,}closed')]

inf_rgx = [re.compile(r'no (\w* ){0,}infiltrat\w+|without (\w*(?!defect) ){0,}infiltrat'),
           re.compile(r'infiltrat\w+ ((?!nearly)(?!almost)\w* ){0,}healed|healed (\w* ){0,}infiltrat\w+'),
           re.compile(r'infiltrat\w+ ((?!almost)(?!mostly)\w* ){0,}resolved|resolved (\w* ){0,}infiltrat'),
           re.compile(r'punctate( \w*){0,}\/?infiltrat\w+'),re.compile(r'punctate (\w* ){0,}infiltrat\w+'),
           re.compile(r'pinpoint (\w* ){0,}infiltrat\w+')]

## match numerical measurements
re_float = re.compile(r'\d*\.?\/?\:?\-?\/?\d+\/?\d*\%?')





def measure_or_not(sents,rgx,kwd):
    '''
        input:
            sents: string of sentences
            rgx: regex pattern of 0 cases
            kwd: keyword you are looking for, ex: defect, infiltrate
        output: list of binarys indicating whether each segment has related measures to the keyword or not
    '''
    sent_lst = split_para(sents)
    sent_lst = apply_f(sent_lst,rm_post_culture)
    measure = list(map(lambda x: size_determ(rgx,kwd,x), sent_lst))
    measure = list(map(lambda x: 1 if x >=1 else 0, measure))
    return measure

def measure(sents, kwd, rgx,re_float=re_float):
    '''
        input:
            sents: string of sentences
            kwd: keyword you are looking for, ex: defect, infiltrate
            rgx: regex pattern of 0 cases
            re_float: regex matching any numerical tokens
        output: list of lists for the numerical measures found at segment level
    '''
    sent_lst = split_para(sents)
    sent_lst = apply_f(sent_lst,rm_post_culture) ## splitted segments
    measure_01 = measure_or_not(sents,rgx,kwd) ## binary indicator

    size = apply_f(sent_lst, lambda t: re.findall(re_float, t)) ## all numerical tokens
    size = no_measures(rgx, sent_lst, size) ## find 0 cases and assign measurement as 0
    new_size=[]
    for i in range(0,len(size)):
        new_size.append(match_numbers(size[i], measure_01[i]))

    results = list(map(lambda x:rm_specialfmt(x), new_size)) ## remove unwanted numerical tokens
    results = list(filter(None,results))
    ## take the first 2 measures if found multiple
    results = list(map(lambda x: x[:2] if len(x)>=2 else [x[0],x[0]], results))
    return results

def measure_max(sents, kwd, rgx,re_float=re_float):
    '''
        input:
            sents: string of sentences
            kwd: keyword you are looking for, ex: defect, infiltrate
            rgx: regex pattern of 0 cases
            re_float: regex matching any numerical tokens
        output: dictionary for the numerical measures related to kwd at encounter level
    '''
    measures = measure(sents, kwd,rgx, re_float=re_float)
    if measures == None:
        return {kwd: (None,None)}
    measures = apply_f(measures,eval)
    if len(measures)==1 and type(measures[0])==list:
        return {kwd: tuple(measures[0])}
    elif len(measures)>=2:
        mean = list(map(stat.mean,measures))
        idx = mean.index(max(mean))
        return {kwd: tuple(measures[idx])}
    else:
        return {kwd: (None,None)}
    



def measure_size(sents,re_float=re_float, rgx=[ed_rgx,inf_rgx]):
    '''
        input:
            sents: string of sentences
            rgx: regex pattern of 0 cases
            re_float: regex matching any numerical tokens
        output: dictionary of valid measurements at encounter level, both defect and infiltrate
    '''
    ed_rgx = rgx[0]
    inf_rgx = rgx[1]
    ed_size=measure_max(sents, 'defect',re_float=re_float, rgx=ed_rgx).get('defect')
    st_size=measure_max(sents, 'stain',re_float=re_float, rgx=ed_rgx).get('stain')
    if ed_size==(None,None) and st_size!=(None,None):
        ed_size=st_size
    inf_size=measure_max(sents, 'infiltrat',re_float=re_float, rgx=inf_rgx).get('infiltrat')
    uc_size=measure_max(sents, 'ulcer',re_float=re_float, rgx=inf_rgx).get('ulcer')
    if inf_size==(None,None) and uc_size!=(None,None):
        inf_size=uc_size
    sents = sents.replace("'","")
    if re.findall(r'defect[\s\w*]*\,? [\w*\s]*with[\s\w*]* infiltrate', sents)!=[] and ed_size==(None,None) and inf_size!=(None,None):
        ed_size=inf_size
    if re.findall(r'defect[\s\w*]*\,? [\w*\s]*with[\s\w*]* infiltrate', sents)!=[] and ed_size!=(None,None) and inf_size==(None,None):
        inf_size=ed_size
    if re.findall(r'infiltrate[\s\w*]*\,? [\w*\s]*with[\s\w*]* defect', sents)!=[] and ed_size!=(None,None) and inf_size==(None,None):
        inf_size=ed_size
    if re.findall(r'infiltrate[\s\w*]*\,? [\w*\s]*with[\s\w*]* defect', sents)!=[] and ed_size==(None,None) and inf_size!=(None,None):
        ed_size=inf_size
    return{'defect':ed_size, 'infiltrate': inf_size}


## example:

sent = '''3+ superficial punctate keratitis diffusely over graft; epithelial defect measuring 4.8 mm vertically x 2.1 mm horizontally with no stromal infiltrate underlying defect; small epithelial defect over central axis measuring x 0.9 mm vertical x 1.0 mm horizontal ; moderate diffuse stromal edema with descemet folds ; superficial stuck-on-appearing plaque-like opacities diffusely on graft with hazy borders, no significant corneal thinning'''


print(measure_size(sent))
print(measure_max(sent, 'infiltrate',inf_rgx,re_float=re_float))
print(measure_max(sent, 'defect',ed_rgx,re_float=re_float))

## on the whole dataframe
inp_path = 'df_corrected.csv'
out_path = 'measured.csv'
left_eye_col = 'corrected_left'    ## output smart text column after spelling correction: left eyes
right_eye_col = 'corrected_right'    ## output smart text column after spelling correction: right eyes

#dat=open(dat_inp,errors='ignore')
#dat=dat.read()
#test = StringIO(dat)
df = pd.read_csv(inp_path, sep=",",na_values=" ")

df['description'] = df[left_eye_col] + '. ' + df[right_eye_col]
df['description'] = pd.Series([re.sub(r'(\d+\:?\d*)-(\d+\:?\d*)','\1 - \2', x) for x in df.description.values.tolist()], index = df.index)
measures = [measure_size(x) for x in df.description.values.tolist()]

df['ed_measure1'] = pd.Series([x['defect'][0] for x in measures])
df['ed_measure2'] = pd.Series([x['defect'][1] for x in measures])
df['si_measure1'] = pd.Series([x['infiltrate'][0] for x in measures])
df['si_measure2'] = pd.Series([x['infiltrate'][1] for x in measures])

df.to_csv(out_path)
