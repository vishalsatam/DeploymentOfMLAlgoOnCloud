import pandas as pd
import numpy as np
import datetime
import time
import logging
import os
from sklearn.cross_validation import train_test_split
from imblearn.under_sampling import RandomUnderSampler

def convertDataTypeOrig(sampog_2005):
    sampog_2005[['CREDIT_SCORE']]=sampog_2005[['CREDIT_SCORE']].astype(str).astype('int64')
    sampog_2005[['MSA']]=sampog_2005[['MSA']].astype('int64')
    sampog_2005[['NUM_UNITS']]=sampog_2005[['NUM_UNITS']].astype('int64')
    sampog_2005[['OG_UPB']]=sampog_2005[['OG_UPB']].astype('int64')
    sampog_2005[['POSTALCODE']]=sampog_2005[['POSTALCODE']].astype('int64')
    sampog_2005[['NUM_BORROWERS']]=sampog_2005[['NUM_BORROWERS']].astype('int64')
    sampog_2005[['OG_LOANTERM']]=sampog_2005[['OG_LOANTERM']].astype('int64')
    
    sampog_2005[['MI_PERCENT']]=sampog_2005[['MI_PERCENT']].astype(str).astype(float)
    sampog_2005[['OG_CLTV']]=sampog_2005[['OG_CLTV']].astype(str).astype(float)
    sampog_2005[['OG_DTI']]=sampog_2005[['OG_DTI']].astype(str).astype(float)
    sampog_2005[['OG_LTV']]=sampog_2005[['OG_LTV']].astype(str).astype(float)
    sampog_2005[['OG_INTERESTRATE']]=sampog_2005[['OG_INTERESTRATE']].astype(str).astype(float)
    #'MI_PERCENT','OG_CLTV','OG_DTI','OG_LTV','OG_INTERESTRATE'
    
    sampog_2005[['FIRST_HOME_BUYER_FLAG']]=sampog_2005[['FIRST_HOME_BUYER_FLAG']].astype(str)
    sampog_2005[['OCCUPANCY_STATS']]=sampog_2005[['OCCUPANCY_STATS']].astype(str)
    sampog_2005[['CHANNEL']]=sampog_2005[['CHANNEL']].astype(str)
    sampog_2005[['PPM_FLAG']]=sampog_2005[['PPM_FLAG']].astype(str)
    sampog_2005[['PRODUCT_TYPE']]=sampog_2005[['PRODUCT_TYPE']].astype(str)
    sampog_2005[['PROP_STATE']]=sampog_2005[['PROP_STATE']].astype(str)
    sampog_2005[['PROP_TYPE']]=sampog_2005[['PROP_TYPE']].astype(str)
    sampog_2005[['LOAN_SEQ_NO']]=sampog_2005[['LOAN_SEQ_NO']].astype(str)
    sampog_2005[['LOAN_PURPOSE']]=sampog_2005[['LOAN_PURPOSE']].astype(str)
    sampog_2005[['SELLER_NAME']]=sampog_2005[['SELLER_NAME']].astype(str)
    sampog_2005[['SERVICE_NAME']]=sampog_2005[['SERVICE_NAME']].astype(str)
    sampog_2005[['LOAN_SEQ_NO']]=sampog_2005[['LOAN_SEQ_NO']].astype(str)
    
    # date type
    sampog_2005['FIRST_PAY_DATE'] =  pd.to_datetime(sampog_2005['FIRST_PAY_DATE'], format='%Y%M')
    sampog_2005['MATURITY_DATE']=pd.to_datetime(sampog_2005['MATURITY_DATE'], format='%Y%M')
    return sampog_2005


def replaceBlanksOrig(sampog_2005):
    sampog_2005.CREDIT_SCORE.replace(to_replace='   ',value='0',inplace=True)
    sampog_2005.MSA.replace('     ','0',inplace=True)
    sampog_2005.MI_PERCENT.replace('   ','0',inplace=True)
    sampog_2005.NUM_UNITS.replace(' ','0',inplace=True)
    sampog_2005.OCCUPANCY_STATS.replace(' ','X',inplace=True)
    sampog_2005.OG_DTI.replace('   ','0',inplace=True)
    sampog_2005.OG_LTV.replace('   ','0',inplace=True)
    sampog_2005.CHANNEL.replace(' ','X',inplace=True)
    sampog_2005.PPM_FLAG.replace(' ','X',inplace=True)
    sampog_2005.PROP_TYPE.replace('  ','XX',inplace=True)
    sampog_2005.POSTALCODE.replace('     ','0',inplace=True)
    sampog_2005.LOAN_PURPOSE.replace(' ','X',inplace=True)
    sampog_2005.NUM_BORROWERS.replace('  ','00',inplace=True)
    return sampog_2005
    
def fillNullOrig(sampog_2005):
    sampog_2005.CREDIT_SCORE.replace(regex="[^0-9]+",value="0",inplace=True)
    sampog_2005['CREDIT_SCORE']=sampog_2005['CREDIT_SCORE'].fillna(0)
    sampog_2005['FIRST_PAY_DATE']=sampog_2005['FIRST_PAY_DATE'].fillna('170001')
    sampog_2005['FIRST_HOME_BUYER_FLAG']=sampog_2005['FIRST_HOME_BUYER_FLAG'].fillna('X')
    sampog_2005['MATURITY_DATE']=sampog_2005['MATURITY_DATE'].fillna('170001')
    sampog_2005['MSA']=sampog_2005['MSA'].fillna(0)
    sampog_2005['MI_PERCENT']=sampog_2005['MI_PERCENT'].fillna(0)
    sampog_2005['NUM_UNITS']=sampog_2005['NUM_UNITS'].fillna(0)
    sampog_2005['OCCUPANCY_STATS']=sampog_2005['OCCUPANCY_STATS'].fillna('X')
    sampog_2005['OG_CLTV']=sampog_2005['OG_CLTV'].fillna(0)
    sampog_2005['OG_DTI']=sampog_2005['OG_DTI'].fillna(0)
    sampog_2005['OG_UPB']=sampog_2005['OG_UPB'].fillna(0)
    sampog_2005['OG_INTERESTRATE']=sampog_2005['OG_INTERESTRATE'].fillna(0)
    sampog_2005['OG_LTV']=sampog_2005['OG_LTV'].fillna(0)
    sampog_2005['POSTALCODE']=sampog_2005['POSTALCODE'].fillna(0)
    sampog_2005['NUM_BORROWERS']=sampog_2005['NUM_BORROWERS'].fillna(0)
    sampog_2005['CHANNEL']=sampog_2005['CHANNEL'].fillna('X')
    sampog_2005['PPM_FLAG']=sampog_2005['PPM_FLAG'].fillna('X')
    sampog_2005['PRODUCT_TYPE']=sampog_2005['PRODUCT_TYPE'].fillna('X')
    sampog_2005['PROP_STATE']=sampog_2005['PROP_STATE'].fillna('X')
    sampog_2005['PROP_TYPE']=sampog_2005['PROP_TYPE'].fillna('X')
    sampog_2005['LOAN_PURPOSE']=sampog_2005['LOAN_PURPOSE'].fillna('X')
    sampog_2005['OG_LOANTERM']=sampog_2005['OG_LOANTERM'].fillna(0)
    sampog_2005['SELLER_NAME']=sampog_2005['SELLER_NAME'].fillna('X')
    sampog_2005['SERVICE_NAME']=sampog_2005['SERVICE_NAME'].fillna('X')
    return sampog_2005

def fillNullPerf(performance_all):
    performance_all=performance_all[performance_all['LOAN_SEQ_NO'].notnull()]
    performance_all['CUR_ACT_UPB']=performance_all['CUR_ACT_UPB'].fillna(0)
    performance_all['CUR_LOAN_DELQ_STAT']=performance_all['CUR_LOAN_DELQ_STAT'].fillna('XX')
    performance_all['CUR_LOAN_DELQ_STAT'] = [ 999 if x=='R' else x for x in (performance_all['CUR_LOAN_DELQ_STAT'].apply(lambda x: x))]
    performance_all['CUR_LOAN_DELQ_STAT'] = [ 0 if x=='XX' else x for x in (performance_all['CUR_LOAN_DELQ_STAT'].apply(lambda x: x))]
    performance_all['LOAN_AGE']=performance_all['LOAN_AGE'].fillna(0)
    performance_all['MONTHS_LEGAL_MATURITY']=performance_all['MONTHS_LEGAL_MATURITY'].fillna(0)
    performance_all['CURR_INTERESTRATE']=performance_all['CURR_INTERESTRATE'].fillna(0)
    performance_all['MONTHLY_REPORT_PERIOD']=performance_all['MONTHLY_REPORT_PERIOD'].fillna('170001')
    performance_all['CURR_DEF_UPB']=performance_all['CURR_DEF_UPB'].fillna(0)
    performance_all['MOD_COST']=performance_all['MOD_COST'].fillna(0)
    performance_all['REPURCHASE_FLAG']=performance_all['REPURCHASE_FLAG'].fillna('X')
    performance_all['ZERO_BAL_EFF_DATE']=performance_all['ZERO_BAL_EFF_DATE'].fillna('170001')
    performance_all['ZERO_BAL_CODE']=performance_all['ZERO_BAL_CODE'].fillna(00)
    performance_all['DDLPI']=performance_all['DDLPI'].fillna('170001')
    performance_all['NET_SALES_PROCEDS']=performance_all['NET_SALES_PROCEDS'].fillna('U')
    performance_all['MI_RECOVERIES']=performance_all['MI_RECOVERIES'].fillna(0)
    performance_all['EXPENSES']=performance_all['EXPENSES'].fillna(0)
    performance_all['LEGAL_COSTS']=performance_all['LEGAL_COSTS'].fillna(0)
    performance_all['MNTC_PRES_COST']=performance_all['MNTC_PRES_COST'].fillna(0)
    performance_all['TAX_INSUR']=performance_all['TAX_INSUR'].fillna(0)
    performance_all['MIS_EXPENSES']=performance_all['MIS_EXPENSES'].fillna(0)
    performance_all['ACT_LOSS_CALC']=performance_all['ACT_LOSS_CALC'].fillna(0)
    performance_all['NON_MI_RECOV']=performance_all['NON_MI_RECOV'].fillna(0)
    performance_all['MOD_FLAG']=performance_all['MOD_FLAG'].fillna('N') # yes or no value.
    return performance_all

def convertTypesPerf(performance_all):
    performance_all["CUR_LOAN_DELQ_STAT"] = performance_all["CUR_LOAN_DELQ_STAT"].apply(np.int64)
    performance_all["MONTHLY_REPORT_PERIOD"] = performance_all["MONTHLY_REPORT_PERIOD"].apply(np.int64)
    performance_all["DDLPI"] = performance_all["DDLPI"].apply(np.int64)
    performance_all['MONTHLY_REPORT_PERIOD'] =  pd.to_datetime(performance_all['MONTHLY_REPORT_PERIOD'], format='%Y%M')
    performance_all['DDLPI']=pd.to_datetime(performance_all['DDLPI'], format='%Y%M')
    performance_all['MONTHLY_REPORT_PERIOD']=performance_all['MONTHLY_REPORT_PERIOD'].dt.date
    performance_all['DDLPI']=performance_all['DDLPI'].dt.date
    performance_all["ZERO_BAL_EFF_DATE"] = performance_all["ZERO_BAL_EFF_DATE"].apply(np.int64)
    performance_all['ZERO_BAL_EFF_DATE']=pd.to_datetime(performance_all['ZERO_BAL_EFF_DATE'], format='%Y%M')
    performance_all['ZERO_BAL_EFF_DATE']=performance_all['ZERO_BAL_EFF_DATE'].dt.date
    return performance_all

def cleanOrig(filepath):
    origcollist=['CREDIT_SCORE','FIRST_PAY_DATE','FIRST_HOME_BUYER_FLAG','MATURITY_DATE','MSA','MI_PERCENT','NUM_UNITS','OCCUPANCY_STATS','OG_CLTV','OG_DTI','OG_UPB','OG_LTV','OG_INTERESTRATE','CHANNEL','PPM_FLAG','PRODUCT_TYPE','PROP_STATE','PROP_TYPE','POSTALCODE','LOAN_SEQ_NO','LOAN_PURPOSE','OG_LOANTERM','NUM_BORROWERS','SELLER_NAME','SERVICE_NAME','SUPER_CONFORMING_FLAG']
    df_orig=pd.read_csv(filepath,delimiter="|",names=origcollist,low_memory=False)
    df_orig=df_orig[['CREDIT_SCORE','FIRST_PAY_DATE','FIRST_HOME_BUYER_FLAG','MATURITY_DATE','MSA','MI_PERCENT','NUM_UNITS','OCCUPANCY_STATS','OG_CLTV','OG_DTI','OG_UPB','OG_LTV','OG_INTERESTRATE','CHANNEL','PPM_FLAG','PRODUCT_TYPE','PROP_STATE','PROP_TYPE','POSTALCODE','LOAN_SEQ_NO','LOAN_PURPOSE','OG_LOANTERM','NUM_BORROWERS','SELLER_NAME','SERVICE_NAME']]
   
    #cleaning code
    df_orig=replaceBlanksOrig(df_orig)
    df_orig=fillNullOrig(df_orig)
    df_orig=convertDataTypeOrig(df_orig)
    #end of cleaning code
    return df_orig
def cleanAndMergeOrig(downpath,origcombinefilelist):
    header=True
    with open(downpath, 'w',encoding='utf-8',newline="") as file:
        for filepath in origcombinefilelist:
            st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
            print("Summarizing : "+st + ":"+filepath)
            logging.info("Summarizing : "+st + ":"+filepath)
            df_orig=cleanOrig(filepath)
            df_orig['OG_YEAR'] = ['19'+x if x=='99' else '20'+x for x in (df_orig['LOAN_SEQ_NO'].apply(lambda x: x[2:4]))]
            df_orig['OG_QUARTER'] = [x for x in (df_orig['LOAN_SEQ_NO'].apply(lambda x: x[4:6]))]
            df_orig['OG_QUARTERYEAR'] = df_orig.apply(lambda x:'%s%s' % (x['OG_QUARTER'],x['OG_YEAR']),axis=1)
            if header is True:    
                df_orig.to_csv(file, mode='a', header=True,index=False)
                header = False
            else:   
                df_orig.to_csv(file, mode='a', header=False,index=False)
            st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
            print("End of Summarizing : "+st + ":"+filepath)
            os.remove(filepath) 

def SUMM_CUR_ACT_UPB(groupdf):
    return {'MIN_CUR_ACT_UPB': groupdf.min(), 'MAX_CUR_ACT_UPB': groupdf.max()}
def SUMM_CUR_LOAN_DELQ_STAT(groupdf):
    return {'MIN_CUR_LOAN_DELQ_STAT': groupdf.min(), 'MAX_CUR_LOAN_DELQ_STAT': groupdf.max()}
def SUMM_MONTHLY_REPORT_PERIOD(groupdf):
    return {'MIN_MONTHLY_REPORT_PERIOD': groupdf.min(), 'MAX_MONTHLY_REPORT_PERIOD': groupdf.max()}
def SUMM_LOAN_AGE(groupdf):
    return {'MIN_LOAN_AGE': groupdf.min(), 'MAX_LOAN_AGE': groupdf.max()}
def SUMM_MONTHS_LEGAL_MATURITY(groupdf):
    return {'MIN_MONTHS_LEGAL_MATURITY': groupdf.min(), 'MAX_MONTHS_LEGAL_MATURITY': groupdf.max()}
def SUMM_REPURCHASE_FLAG(group):
    if 'Y' in group.unique():
        return {'REPURCHASED':'Y'}
    else:
        return {'REPURCHASED':'N'}
def SUMM_ZERO_BAL_CODE(groupdf):
    return {'MIN_ZERO_BAL_CODE': groupdf.min(), 'MAX_ZERO_BAL_CODE': groupdf.max()}
def SUMM_CURR_INTERESTRATE(groupdf):
    return {'MIN_CURR_INTERESTRATE': groupdf.min(), 'MAX_CURR_INTERESTRATE': groupdf.max()}
def SUMM_CURR_DEF_UPB(groupdf):
    return {'MIN_CURR_DEF_UPB': groupdf.min(), 'MAX_CURR_DEF_UPB': groupdf.max()}
def SUMM_MI_RECOVERIES(groupdf):
    return {'MIN_MI_RECOVERIES': groupdf.min(), 'MAX_MI_RECOVERIES': groupdf.max()}
def SUMM_NON_MI_RECOV(groupdf):
    return {'MIN_NON_MI_RECOV': groupdf.min(), 'MAX_NON_MI_RECOV': groupdf.max()}
def SUMM_EXPENSES(groupdf):
    return {'MIN_EXPENSES': groupdf.min(), 'MAX_EXPENSES': groupdf.max()}
def SUMM_LEGAL_COSTS(groupdf):
    return {'MIN_LEGAL_COSTS': groupdf.min(), 'MAX_LEGAL_COSTS': groupdf.max()}
def SUMM_TAX_INSUR(groupdf):
    return {'MIN_TAX_INSUR': groupdf.min(), 'MAX_TAX_INSUR': groupdf.max()}
def SUMM_ACT_LOSS_CALC(groupdf):
    return {'MIN_ACT_LOSS_CALC': groupdf.min(), 'MAX_ACT_LOSS_CALC': groupdf.max()}
def SUMM_MOD_COST(groupdf):
    return {'MIN_MOD_COST': groupdf.min(), 'MAX_MOD_COST': groupdf.max()}

def summarizeperformancefiles(df_summary):
    summary_df=pd.DataFrame()
    summary_df['LOAN_SEQ_NO']=df_summary['LOAN_SEQ_NO'].drop_duplicates()
    summary_df=summary_df.join((df_summary['CUR_ACT_UPB'].groupby(df_summary['LOAN_SEQ_NO']).apply(SUMM_CUR_ACT_UPB).unstack()),on='LOAN_SEQ_NO')
    summary_df=summary_df.join((df_summary['CUR_LOAN_DELQ_STAT'].groupby(df_summary['LOAN_SEQ_NO']).apply(SUMM_CUR_LOAN_DELQ_STAT).unstack()),on='LOAN_SEQ_NO')
    summary_df=summary_df.join((df_summary['CURR_INTERESTRATE'].groupby(df_summary['LOAN_SEQ_NO']).apply(SUMM_CURR_INTERESTRATE).unstack()),on='LOAN_SEQ_NO')
    summary_df=summary_df.join((df_summary['MONTHLY_REPORT_PERIOD'].groupby(df_summary['LOAN_SEQ_NO']).apply(SUMM_MONTHLY_REPORT_PERIOD).unstack()),on='LOAN_SEQ_NO')
    summary_df=summary_df.join((df_summary['LOAN_AGE'].groupby(df_summary['LOAN_SEQ_NO']).apply(SUMM_LOAN_AGE).unstack()),on='LOAN_SEQ_NO')
    summary_df=summary_df.join((df_summary['MONTHS_LEGAL_MATURITY'].groupby(df_summary['LOAN_SEQ_NO']).apply(SUMM_MONTHS_LEGAL_MATURITY).unstack()),on='LOAN_SEQ_NO')
    summary_df=summary_df.join((df_summary['REPURCHASE_FLAG'].groupby(df_summary['LOAN_SEQ_NO']).apply(SUMM_REPURCHASE_FLAG).unstack()),on='LOAN_SEQ_NO')
    summary_df=summary_df.join((df_summary['ZERO_BAL_CODE'].groupby(df_summary['LOAN_SEQ_NO']).apply(SUMM_ZERO_BAL_CODE).unstack()),on='LOAN_SEQ_NO')
    summary_df=summary_df.join((df_summary['CURR_DEF_UPB'].groupby(df_summary['LOAN_SEQ_NO']).apply(SUMM_CURR_DEF_UPB).unstack()),on='LOAN_SEQ_NO')
    summary_df=summary_df.join((df_summary['MI_RECOVERIES'].groupby(df_summary['LOAN_SEQ_NO']).apply(SUMM_MI_RECOVERIES).unstack()),on='LOAN_SEQ_NO')
    summary_df=summary_df.join((df_summary['NON_MI_RECOV'].groupby(df_summary['LOAN_SEQ_NO']).apply(SUMM_NON_MI_RECOV).unstack()),on='LOAN_SEQ_NO')
    summary_df=summary_df.join((df_summary['EXPENSES'].groupby(df_summary['LOAN_SEQ_NO']).apply(SUMM_EXPENSES).unstack()),on='LOAN_SEQ_NO')
    summary_df=summary_df.join((df_summary['LEGAL_COSTS'].groupby(df_summary['LOAN_SEQ_NO']).apply(SUMM_LEGAL_COSTS).unstack()),on='LOAN_SEQ_NO')
    summary_df=summary_df.join((df_summary['TAX_INSUR'].groupby(df_summary['LOAN_SEQ_NO']).apply(SUMM_TAX_INSUR).unstack()),on='LOAN_SEQ_NO')
    summary_df=summary_df.join((df_summary['ACT_LOSS_CALC'].groupby(df_summary['LOAN_SEQ_NO']).apply(SUMM_ACT_LOSS_CALC).unstack()),on='LOAN_SEQ_NO')
    summary_df=summary_df.join((df_summary['MOD_COST'].groupby(df_summary['LOAN_SEQ_NO']).apply(SUMM_MOD_COST).unstack()),on='LOAN_SEQ_NO')
    return summary_df

   
def cleanPerf(filepath):
    summcollist=['LOAN_SEQ_NO','MONTHLY_REPORT_PERIOD','CUR_ACT_UPB','CUR_LOAN_DELQ_STAT','LOAN_AGE','MONTHS_LEGAL_MATURITY','REPURCHASE_FLAG','MOD_FLAG','ZERO_BAL_CODE','ZERO_BAL_EFF_DATE','CURR_INTERESTRATE','CURR_DEF_UPB','DDLPI','MI_RECOVERIES','NET_SALES_PROCEDS','NON_MI_RECOV','EXPENSES','LEGAL_COSTS','MNTC_PRES_COST','TAX_INSUR','MIS_EXPENSES','ACT_LOSS_CALC','MOD_COST']
    df_summary=pd.read_csv(filepath,delimiter="|",names=summcollist,low_memory=False)
    #cleaning code
    df_summary=(fillNullPerf(df_summary))
    df_summary=(convertTypesPerf(df_summary))

def convertTypesPerfNoSummarize(performance_all):
    performance_all["CUR_LOAN_DELQ_STAT"] = performance_all["CUR_LOAN_DELQ_STAT"].apply(np.int64)
    performance_all["MONTHLY_REPORT_PERIOD"] = performance_all["MONTHLY_REPORT_PERIOD"].apply(np.int64)
    performance_all['MONTHLY_REPORT_PERIOD'] =  pd.to_datetime(performance_all['MONTHLY_REPORT_PERIOD'], format='%Y%M')
    performance_all['MONTHLY_REPORT_PERIOD']=performance_all['MONTHLY_REPORT_PERIOD'].dt.date
    return performance_all

def fillNullPerfNoSumarize(performance_all):
    performance_all=performance_all[performance_all['LOAN_SEQ_NO'].notnull()]
    performance_all['CUR_ACT_UPB']=performance_all['CUR_ACT_UPB'].fillna(0)
    performance_all['CUR_LOAN_DELQ_STAT']=performance_all['CUR_LOAN_DELQ_STAT'].fillna('XX')
    performance_all['CUR_LOAN_DELQ_STAT'] = [ 999 if x=='R' else x for x in (performance_all['CUR_LOAN_DELQ_STAT'].apply(lambda x: x))]
    performance_all['CUR_LOAN_DELQ_STAT'] = [ 0 if x=='XX' else x for x in (performance_all['CUR_LOAN_DELQ_STAT'].apply(lambda x: x))]
    performance_all['LOAN_AGE']=performance_all['LOAN_AGE'].fillna(0)
    performance_all['MONTHS_LEGAL_MATURITY']=performance_all['MONTHS_LEGAL_MATURITY'].fillna(0)
    performance_all['CURR_INTERESTRATE']=performance_all['CURR_INTERESTRATE'].fillna(0)
    performance_all['MONTHLY_REPORT_PERIOD']=performance_all['MONTHLY_REPORT_PERIOD'].fillna('170001')
    performance_all['CURR_DEF_UPB']=performance_all['CURR_DEF_UPB'].fillna(0)
    return performance_all


def cleanPerfNoSummarize(filepath):
    summcollist=['LOAN_SEQ_NO','MONTHLY_REPORT_PERIOD','CUR_ACT_UPB','CUR_LOAN_DELQ_STAT','LOAN_AGE','MONTHS_LEGAL_MATURITY','REPURCHASE_FLAG','MOD_FLAG','ZERO_BAL_CODE','ZERO_BAL_EFF_DATE','CURR_INTERESTRATE','CURR_DEF_UPB','DDLPI','MI_RECOVERIES','NET_SALES_PROCEDS','NON_MI_RECOV','EXPENSES','LEGAL_COSTS','MNTC_PRES_COST','TAX_INSUR','MIS_EXPENSES','ACT_LOSS_CALC','MOD_COST']
    df_summary=pd.read_csv(filepath,delimiter="|",names=summcollist,low_memory=False)
    #cleaning code
    df_summary=(fillNullPerfNoSumarize(df_summary))
    df_summary=(convertTypesPerfNoSummarize(df_summary))
    df_summary['CUR_LOAN_DELQ_STAT']=df_summary['CUR_LOAN_DELQ_STAT'].apply(lambda x: label_Delinquency(x))
    df_summary=df_summary[['MONTHLY_REPORT_PERIOD','CUR_ACT_UPB','CUR_LOAN_DELQ_STAT','LOAN_AGE','MONTHS_LEGAL_MATURITY','CURR_INTERESTRATE','CURR_DEF_UPB']]

    
    #df_summary=summarizeperformancefiles(df_summary)
    #end of cleaning code
    return df_summary




def cleanAndMergePerfNoSummarization(downpath,perfcombinefilelist,trainoutputpath,testoutputpath):
    header=True
    with open(downpath, 'w',encoding='utf-8',newline="") as file:
        for filepath in perfcombinefilelist:
            st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
            print("Summarizing : "+st + ":"+filepath)
            df_summary=cleanPerfNoSummarize(filepath)
            if header is True:
                df_summary.to_csv(file, mode='a', header=True,index=False)
                header = False
            else:
                df_summary.to_csv(file, mode='a', header=False,index=False)
            st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
            print("End of Summarizing : "+st + ":"+filepath)
            os.remove(filepath)

def cleanAndMergePerf(downpath,perfcombinefilelist):
    header=True
    with open(downpath, 'w',encoding='utf-8',newline="") as file:
        for filepath in perfcombinefilelist:
            st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
            print("Summarizing : "+st + ":"+filepath)
            df_summary=cleanPerf(filepath)
            df_summary=summarizeperformancefiles(df_summary)
            if header is True:
                df_summary.to_csv(file, mode='a', header=True,index=False)
                header = False
            else:
                df_summary.to_csv(file, mode='a', header=False,index=False)
            st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')
            print("End of Summarizing : "+st + ":"+filepath)
            os.remove(filepath)

def cleanOrigPrediction(filepath):
    origcollist=['CREDIT_SCORE','FIRST_PAY_DATE','FIRST_HOME_BUYER_FLAG','MATURITY_DATE','MSA','MI_PERCENT','NUM_UNITS','OCCUPANCY_STATS','OG_CLTV','OG_DTI','OG_UPB','OG_LTV','OG_INTERESTRATE','CHANNEL','PPM_FLAG','PRODUCT_TYPE','PROP_STATE','PROP_TYPE','POSTALCODE','LOAN_SEQ_NO','LOAN_PURPOSE','OG_LOANTERM','NUM_BORROWERS','SELLER_NAME','SERVICE_NAME','SUPER_CONFORMING_FLAG']
    df_orig=pd.read_csv(filepath,delimiter="|",names=origcollist,low_memory=False)
    df_orig=df_orig[['CREDIT_SCORE','FIRST_PAY_DATE','FIRST_HOME_BUYER_FLAG','MATURITY_DATE','MSA','MI_PERCENT','NUM_UNITS','OCCUPANCY_STATS','OG_CLTV','OG_DTI','OG_UPB','OG_LTV','OG_INTERESTRATE','CHANNEL','PPM_FLAG','PRODUCT_TYPE','PROP_STATE','PROP_TYPE','POSTALCODE','LOAN_SEQ_NO','LOAN_PURPOSE','OG_LOANTERM','NUM_BORROWERS','SELLER_NAME','SERVICE_NAME']]
    #cleaning code
    df_orig=replaceBlanksOrig(df_orig)
    df_orig=fillNullOrig(df_orig)
    df_orig=convertDataTypeOrig(df_orig)
    #end of cleaning code
    return df_orig         
            
            
def fillNullPerfClassification(performance_all):
    performance_all=performance_all[performance_all['LOAN_SEQ_NO'].notnull()]
    performance_all['CUR_ACT_UPB']=performance_all['CUR_ACT_UPB'].fillna(0)
    performance_all['CUR_LOAN_DELQ_STAT']=performance_all['CUR_LOAN_DELQ_STAT'].fillna('XX')
    performance_all['CUR_LOAN_DELQ_STAT'] = [ 999 if x=='R' else x for x in (performance_all['CUR_LOAN_DELQ_STAT'].apply(lambda x: x))]
    performance_all['CUR_LOAN_DELQ_STAT'] = [ 0 if x=='XX' else x for x in (performance_all['CUR_LOAN_DELQ_STAT'].apply(lambda x: x))]
    performance_all['LOAN_AGE']=performance_all['LOAN_AGE'].fillna(0)
    performance_all['MONTHS_LEGAL_MATURITY']=performance_all['MONTHS_LEGAL_MATURITY'].fillna(0)
    performance_all['CURR_INTERESTRATE']=performance_all['CURR_INTERESTRATE'].fillna(0)
    performance_all['MONTHLY_REPORT_PERIOD']=performance_all['MONTHLY_REPORT_PERIOD'].fillna('170001')
    performance_all['CURR_DEF_UPB']=performance_all['CURR_DEF_UPB'].fillna(0)

    return performance_all

def label_Delinquency(x):
    if x == 0:
        return '0'
    else:
        return '1'

def convertTypesPerfClassification(performance_all):
    performance_all["CUR_LOAN_DELQ_STAT"] = performance_all["CUR_LOAN_DELQ_STAT"].apply(np.int64)
    performance_all['DELINQUENT']=performance_all['CUR_LOAN_DELQ_STAT'].apply(lambda x: label_Delinquency(x))
    #performance_all["MONTHLY_REPORT_PERIOD"] = performance_all["MONTHLY_REPORT_PERIOD"].apply(np.int64)
    #performance_all['MONTHLY_REPORT_PERIOD'] =  pd.to_datetime(performance_all['MONTHLY_REPORT_PERIOD'], format='%Y%M')
    #performance_all['MONTHLY_REPORT_PERIOD'] =   performance_all['MONTHLY_REPORT_PERIOD'].dt.date

    return performance_all



def cleanPerfClassification(filepath):
    summcollist=['LOAN_SEQ_NO','MONTHLY_REPORT_PERIOD','CUR_ACT_UPB','CUR_LOAN_DELQ_STAT','LOAN_AGE','MONTHS_LEGAL_MATURITY','REPURCHASE_FLAG','MOD_FLAG','ZERO_BAL_CODE','ZERO_BAL_EFF_DATE','CURR_INTERESTRATE','CURR_DEF_UPB','DDLPI','MI_RECOVERIES','NET_SALES_PROCEDS','NON_MI_RECOV','EXPENSES','LEGAL_COSTS','MNTC_PRES_COST','TAX_INSUR','MIS_EXPENSES','ACT_LOSS_CALC','MOD_COST']
    df_summary=pd.read_csv(filepath,delimiter="|",names=summcollist)
    #cleaning code
    df_summary=df_summary[['LOAN_SEQ_NO','MONTHLY_REPORT_PERIOD','CUR_ACT_UPB','CUR_LOAN_DELQ_STAT','LOAN_AGE','MONTHS_LEGAL_MATURITY','CURR_INTERESTRATE','CURR_DEF_UPB']]
    df_summary=(fillNullPerfClassification(df_summary))
    df_summary=(convertTypesPerfClassification(df_summary))
    return df_summary