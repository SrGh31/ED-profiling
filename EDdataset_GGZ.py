import numpy as np
import pandas as pd
import os.path
import sys
fileloc_data='/'.join(os.getcwd().split('/')[0:5])+ '/data/annonymizedDatasets/'

def colsToFloat():
    colRename_dict=colRenaming()
    substrings=['Main-Age','SQ48', 'MHCSF','Lav','DT', 'EDEQ', 'Honos']
    cols_renamed={}
    for sub in substrings:
        res = [i for i in colRename_dict.values() if sub in i]
        cols_renamed[sub]=res
    dtypeDectools=dict(zip(cols_renamed['DT'], ['float', 'float','float', 'float', 'float','float','float','float']))
    dtypeEDEQ=dict(zip(cols_renamed['EDEQ'], ['float', 'float','float', 'float']))
    dtypeHonos=dict(zip(cols_renamed['Honos'], ['float', 'float','float', 'float', 'float','float']))
    dtypeLAV=dict(zip(cols_renamed['Lav'], ['float', 'float','float', 'float']))
    dtypeSQ48=dict(zip(cols_renamed['SQ48'], ['float', 'float','float', 'float', 'float','float','float',
                                                   'float','float','float']))
    dtypeMHCSF=dict(zip(cols_renamed['MHCSF'], ['float', 'float','float', 'float']))
    dtypesToFloat={'DT':dtypeDectools, 'EDEQ':dtypeEDEQ, 'Honos': dtypeHonos, 'Lav':dtypeLAV, 'MHCSF': dtypeMHCSF, 'SQ48':dtypeSQ48}
    return dtypesToFloat

def colsTypeCast(df):
    dtypesToFloat=colsToFloat()
    colRename_dict=colRenaming()
    df.rename(columns=colRename_dict, inplace=True)
    MainCols=['Main-Bsex','Main-Highest_Edu', 'EDtype']
    MainCol_Rename=['Main-Biosex','Main-Education', 'ED_Codes']
    for idx, col in enumerate(MainCols):
        if col in df.columns:
            df[col] = df[col].astype('category')
            df[MainCol_Rename[idx]] = df[col].cat.codes
            df[MainCol_Rename[idx]]=df[MainCol_Rename[idx]].astype(float)
    cols2consider=['Main-Age', 'Main-Biosex', 'Main-Education', 'ED_Codes']
    subscales={}
    #print(df.columns)
    for key, dtype_dict in dtypesToFloat.items():
        res = [i for i in list(df.columns) if key in i]
        #df['BMI'].astype(float)
        #print(key, list(dtype_dict.keys()))
        if key!='EDEQ':
            if len(res)>0:# key in list(df.columns):
                for colname, dtype in dtype_dict.items():
                    df[colname]=df[colname].astype(dtype)
                cols2consider=cols2consider+list(dtype_dict.keys())
                subscales[key]=list(dtype_dict.keys())
        else:
            if len(res)>1:
                for colname, dtype in dtype_dict.items():
                    df[colname]=df[colname].astype(dtype)
                cols2consider=cols2consider+list(dtype_dict.keys())
                subscales[key]=list(dtype_dict.keys())
            else:
                df['EDEQ-Score']=df['EDEQ-Score'].astype(float)
                cols2consider=cols2consider+['EDEQ-Score']
                subscales[key]=['EDEQ-Score']            
    return df[['intid', 'Split', 'EDtype']+cols2consider], cols2consider, subscales


def colRenaming():
    colsDectools=['BMI', 'IND_eerdere_spec_behandeling_zonder_effect', 'aantal_eerdere_trajecten', 'duur_stoornis_in_jaren', 
    'IND_depressie_comorbiditeit','IND_borderline_comorbiditeit', 'IND_ocd_comorbiditeit', 'IND_anders']    
    colsEDEQ=['EDEQ-eating', 'EDEQ-weight', 'EDEQ-bodyshape', 'EDEQ-lines']    
    colsHonos=['Honos-Somscore', 'Honos-Beperkingen', 'Honos-Functioneren', 'Honos-Gedragsproblemen', 
               'Honos-Symptomalogie', 'Honos-SocialeProblemen']    
    colsLAV=['Lav-Neg_Waardering', 'Lav-Gebrek_Vertrouwdheid', 'Lav-Alg_Ontevredenheid', 'Lav-Score']
    colsSQ48=['SQ48-Vijandigheid','SQ48-Agorafobie','SQ48-Angst','SQ48-Depressie', 'SQ48-Cognitieve_Klachten',
        'SQ48-Somatische_Klachten', 'SQ48-Sociale_Fobie', 'SQ48-Vitaliteit_Optimisme', 'SQ48-Werk_Studie', 'SQ48-Score']    
    colsMHCSF=['MHCSF-EmotionWB', 'MHCSF-SocialWB', 'MHCSF-PsychWB', 'MHCSF-Score']   
    colNames=['Main-Age', 'Main-Biosex','Main-Education', 'Main-ED_Codes', 'EDEQ-Score','EDEQ-eating', 'EDEQ-weight', 
    'EDEQ-bodyshape','EDEQ-lines', 'DT-BMI', 'DT-IND_prev_spec_int_wo_eff', 'DT-num_prev_routes', 
    'DT-Disorder_Duration_Yrs', 'DT-IND_depression_CMD','DT-IND_BDL_CMD', 'DT-IND_OCD_CMD', 'DT-IND_others',
    'Lav-Negative_appraisal_body','Lav-Unfamiliarity_with_body', 'Lav-Dissatisfaction_body', 'Lav-Score', 
    'SQ48-Hostility','SQ48-Agoraphobia','SQ48-Anxiety','SQ48-Depression','SQ48-Cognitive_Complaints',
    'SQ48-Somatic_Complaints', 'SQ48-Social_phobia','SQ48-Vitality','SQ48-Work_related_complaints', 'SQ48-Score', 
    'MHCSF-Emotional_Well-being', 'MHCSF-Social_Well-being', 'MHCSF-Psychological_Well-being', 'MHCSF-Score',
    'Honos-Somscore', 'Honos-Limitation','Honos-Functionality', 'Honos-Behaviour_problem', 'Honos-Symptomalogy','Honos-Social_Problems']
    cols2consider=['Main-Age','Main-Biosex', 'Main-Education','ED_Codes','EDEQ-Score']+ colsEDEQ+ colsDectools+ colsLAV+ colsSQ48+\
    colsMHCSF+ colsHonos
    colRename_dict=dict(zip(cols2consider, colNames))
    return colRename_dict

