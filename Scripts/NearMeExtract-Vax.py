# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 23:38:12 2022

@author: Kenneth
"""
#%%
import os
import pandas as pd
import re
import numpy as np

#preprocess/cleaning
def cleanListToDF(list_input,ignore_phrase,columns):
    clean_list=[]
    for each in list_input:
        if each==ignore_phrase:
            continue
        else:
            split_entry=each.split(',')
            
            if len(split_entry)==2:
                
                base=split_entry[1]
                base=base.replace('+','').replace('%','').replace('"','')
                
                clean=[split_entry[0],base]
                clean_list.append(clean)
            else:

                for count,entry in enumerate(split_entry[1:]):
                    if count==0:
                        base=entry
                    else:
                        base=base+entry
                        
                base=base.replace('+','').replace('%','').replace('"','')
                
                clean=[split_entry[0],base]
                clean_list.append(clean)

    clean_df=pd.DataFrame(clean_list,columns=columns)
    
    return clean_df

#feed path, extracts files and compiles dfs
def compileBreakouts(basepath):
    
    files=os.listdir(basepath)
    state_list=[]
    for file in files:
        path=basepath+'\\'+file
        
        df=pd.read_csv(path,delimiter='\n')
        
        df.reset_index(inplace=True)
        
        meta=df.iloc[0].values[0]
        
        reg_state=re.search(',\s(.*)\)',meta)
        
        state=reg_state.group(1)
        
        top_index=df.query('index=="TOP"').index[0]
        rising_index=df.query('index=="RISING"').index[0]
        
        top=df['index'].iloc[top_index:rising_index]
        top_list=top.to_list()
        
        rising=df['index'].iloc[rising_index:]
        rising_list=rising.to_list()
        
        top_df=cleanListToDF(top_list,'TOP',['Query','Value'])
        rising_df=cleanListToDF(rising_list,'RISING',['Query','Value'])
    
        state_list.append([state,top_df,rising_df])
    
    compiled_top_list=[]
    for each in state_list:
        compiled=each[1]
        
        compiled['State']=each[0]
        
        compiled_top_list.append(compiled)
    
    compiled_top_df=pd.concat(compiled_top_list)
    
    compiled_top_df['Value']=compiled_top_df['Value'].astype(int)


    compiled_rising_list=[]
    
    for each in state_list:
        compiled=each[2]
        
        compiled['State']=each[0]
        
        compiled_rising_list.append(compiled)
    
    compiled_rising_df=pd.concat(compiled_rising_list)

    compiled_rising_df_breakouts=compiled_rising_df.query('Value=="Breakout"')
    compiled_rising_df_num=compiled_rising_df.query('Value!="Breakout"')
    
    compiled_rising_df_num['Value']=compiled_rising_df_num['Value'].astype(int)
    
    breakout_num=compiled_rising_df_breakouts.groupby('State').count().reset_index()
    
    compiled_rising_df_breakouts_numeric=compiled_rising_df.replace('Breakout',np.max(compiled_rising_df_num['Value']))
    compiled_rising_df_breakouts_numeric['Value']=compiled_rising_df_breakouts_numeric['Value'].astype(int)
    
    
    return [compiled_top_df,compiled_rising_df_breakouts,compiled_rising_df_breakouts_numeric,breakout_num]

year='2020'
basepath=r'E:\Cartographer\SEOCarto\StateSearchTerms_'+year

[compiled_top_df,compiled_rising_df_breakouts,compiled_rising_df_breakouts_numeric,breakout_num]=compileBreakouts(basepath)

#%%
#import and merge with vaccination data

vax_path=r'E:\Cartographer\SEOCarto\COVID-19_Vaccinations_in_the_United_States_County_2022.csv'

df_vax=pd.read_csv(vax_path)

df_vax_state=df_vax.groupby('Recip_State').sum()

df_vax_state['CalcPer']=df_vax_state['Series_Complete_Yes']/df_vax_state['Census2019']

df_vax_state=df_vax_state.reset_index()

df_vax_state['BoosterPer']=df_vax_state['Booster_Doses']/df_vax_state['Census2019']


df_vax_per=df_vax_state[['Recip_State','CalcPer','BoosterPer']]
#df_vax__per=df_vax_state['Series_Complete_Yes']/df_vax_state['Census2019']

state_abbrev=pd.read_csv(r'E:\Cartographer\SEOCarto\state_abbrev.csv')

df_vax_per_merged=pd.merge(df_vax_per,state_abbrev,how='inner',left_on='Recip_State',right_on='Oct. 1963 - Present')

df_vax_merged_final=pd.merge(df_vax_per_merged,breakout_num,how='inner',left_on='State/Territory',right_on='State')

#%%
#analysis

#boosters
booster_breakout=compiled_rising_df_breakouts[compiled_rising_df_breakouts['Query'].str.contains('booster')]

#covid attn
covid_attn=compiled_rising_df_breakouts_numeric[compiled_rising_df_breakouts_numeric['Query'].str.contains('covid')]
covid_attn_top=compiled_top_df[compiled_top_df['Query'].str.contains('covid')]

covid_vax_attn_top=compiled_top_df[compiled_top_df['Query'].str.contains('vaccine')]
#not representative to do just"covid", some searches don't have that

covid_attn_states=covid_attn.groupby('State').mean().reset_index()
covid_attn_states['Value_Factor']=1/covid_attn_states['Value']#inverse is to help states that have multiple entries with covid

#mayve just a simple count is good enough, ie how many covid queies showed up
covid_attn_counts=covid_attn.groupby('State').count().reset_index()

covid_attn_avg_top=covid_attn_top.groupby('State').mean().reset_index()

#%in order to account for breakouts, set to max value of numeric?

df_Covidattn_merged=pd.merge(df_vax_per_merged,covid_attn_counts,how='inner',left_on='State/Territory',right_on='State')
#df_Covidattn_merged=pd.merge(df_vax_per_merged,covid_attn_counts_top,how='inner',left_on='State/Territory',right_on='State')


df_Covidattn_merged_perCount_avg=df_Covidattn_merged.groupby('Value').mean().reset_index(drop=True)
df_Covidattn_merged_perCount_std=df_Covidattn_merged.groupby('Value').std().reset_index(drop=True)
df_Covidattn_merged_perCount_max=df_Covidattn_merged.groupby('Value').max().reset_index(drop=True)
df_Covidattn_merged_perCount_min=df_Covidattn_merged.groupby('Value').min().reset_index(drop=True)

#
df_Covidattn_merged_2=pd.merge(df_vax_per_merged,covid_attn_avg_top,how='left',left_on='State/Territory',right_on='State')
#df_plot2=df_Covidattn_merged_2[['CalcPer','BoosterPer','Value']]
#df_plot2=df_plot2.query('CalcPer!=0')
#
#plt.plot(df_plot2['Value'],df_plot2['CalcPer'],'x')

#%

#prep data set for sbn and visualize

df_plot=df_Covidattn_merged[['CalcPer','BoosterPer','Value']]
df_plot=df_plot.query('CalcPer!=0')

df_comp=df_plot[['CalcPer','Value']]

df_comp['Type']='FullSeries'
df_comp=df_comp.rename(columns={'CalcPer':'Percent'})
df_comp['Percent']=df_comp['Percent']*100

df_comp2=df_plot[['BoosterPer','Value']]

df_comp2['Type']='Booster'
df_comp2=df_comp2.rename(columns={'BoosterPer':'Percent'})
df_comp2['Percent']=df_comp2['Percent']*100

df_final_plot=pd.concat([df_comp,df_comp2])


import seaborn as sns
import matplotlib.pyplot as plt
#plt.plot(df_Covidattn_merged['Value'],df_Covidattn_merged['CalcPer'],'x')

#sns.set_theme(style="ticks")
#f, ax = plt.subplots(figsize=(7, 6))
#planets = sns.load_dataset("planets")
#tips = sns.load_dataset("tips")

plot=sns.violinplot(x="Value", y="Percent", hue='Type', data=df_final_plot,
            whis=[0, 100], width=.6, palette="vlag")

#ax.xaxis.grid(True)
#ax.set(ylabel="Test")
sns.despine(trim=True, left=True)

plot.set(xlabel="# Rising Google Search Queries with 'Covid', "+year)
plot.set(ylabel="%Vaccination Rate, 2022")
plot.set(title="State Search Attention vs Vaccination Rates")
plt.legend(bbox_to_anchor=(1.3,.55))
#%%
#run linear regression, basic model

from sklearn.linear_model import LinearRegression
import numpy

def runLinReg(x,y):
    reg=LinearRegression().fit(x,y)
    
    return({'Score':reg.score(x,y),'Coef':reg.coef_,'Intercept':reg.intercept_})
    

x=np.array(df_comp['Value']).reshape(-1,1)
y=np.array(df_comp['Percent']).reshape(-1,1)

lin_reg_single=runLinReg(x,y)

print("Score: "+str(lin_reg_single['Score']))



#%%experimental stuff


#multiple factor linear?

#incorporate average top

df_attn_multifact=pd.merge(df_Covidattn_merged,covid_attn_avg_top,how='inner',left_on='State',right_on='State')

df_attn_multifact=df_attn_multifact.rename(columns={'Value_x':'RisingCount','Value_y':'Avg_Top'})

df_attn_multifact=pd.merge(df_attn_multifact,covid_attn_states,how='inner',left_on='State',right_on='State')

x=np.array(df_attn_multifact[['RisingCount','Avg_Top','Value_Factor']])
y=np.array(df_attn_multifact['CalcPer']).reshape(-1,1)


lin_reg_multi=runLinReg(x,y)

#%n-grams?

def ngrams(input, n):
    input = input.split(' ')
    output = []
    for i in range(len(input)-n+1):
        
#        gram=input[i:i+n]
#        gram_combined=
        
        output.append(input[i:i+n])
    
    return output

def find_ngrams(input_list,n):
    output_list=[]
    for each in input_list:
        output_list.extend(ngrams(each,n))
        
    return output_list
    

test=find_ngrams(compiled_rising_df_breakouts_numeric['Query'],2)

df_ngram2=pd.DataFrame(test)
df_ngram2_counts=df_ngram2.value_counts()


