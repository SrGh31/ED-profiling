import os
import pandas as pd
import numpy as np

# Define constants
FLOAT_DTYPES = ['float'] * 10  # Adjust the length as per requirement

# File location
fileloc_data = os.path.join('/', *os.getcwd().split('/')[:5], 'data', 'annonymizedDatasets')


def col_renaming() -> dict:
    """Return a dictionary mapping old column names to new column names."""
    cols_dectools = ['BMI', 'IND_eerdere_spec_behandeling_zonder_effect', 'aantal_eerdere_trajecten', 
                     'duur_stoornis_in_jaren', 'IND_depressie_comorbiditeit', 'IND_borderline_comorbiditeit', 
                     'IND_ocd_comorbiditeit', 'IND_anders']
    cols_edeq = ['EDEQ-eating', 'EDEQ-weight', 'EDEQ-bodyshape', 'EDEQ-lines']
    cols_honos = ['Honos-Somscore', 'Honos-Beperkingen', 'Honos-Functioneren', 'Honos-Gedragsproblemen',
                  'Honos-Symptomalogie', 'Honos-SocialeProblemen']
    cols_lav = ['Lav-Neg_Waardering', 'Lav-Gebrek_Vertrouwdheid', 'Lav-Alg_Ontevredenheid', 'Lav-Score']
    cols_sq48 = ['SQ48-Vijandigheid', 'SQ48-Agorafobie', 'SQ48-Angst', 'SQ48-Depressie', 'SQ48-Cognitieve_Klachten',
                 'SQ48-Somatische_Klachten', 'SQ48-Sociale_Fobie', 'SQ48-Vitaliteit_Optimisme', 'SQ48-Werk_Studie', 'SQ48-Score']
    cols_mhcsf = ['MHCSF-EmotionWB', 'MHCSF-SocialWB', 'MHCSF-PsychWB', 'MHCSF-Score']
    col_names = ['Main-Age', 'Main-Biosex', 'Main-Education', 'Main-ED_Codes', 'EDEQ-Score', 'EDEQ-eating', 'EDEQ-weight',
                 'EDEQ-bodyshape', 'EDEQ-lines', 'DT-BMI', 'DT-IND_prev_spec_int_wo_eff', 'DT-num_prev_routes',
                 'DT-Disorder_Duration_Yrs', 'DT-IND_depression_CMD', 'DT-IND_BDL_CMD', 'DT-IND_OCD_CMD', 'DT-IND_others',
                 'Lav-Negative_appraisal_body', 'Lav-Unfamiliarity_with_body', 'Lav-Dissatisfaction_body', 'Lav-Score',
                 'SQ48-Hostility', 'SQ48-Agoraphobia', 'SQ48-Anxiety', 'SQ48-Depression', 'SQ48-Cognitive_Complaints',
                 'SQ48-Somatic_Complaints', 'SQ48-Social_phobia', 'SQ48-Vitality', 'SQ48-Work_related_complaints', 'SQ48-Score',
                 'MHCSF-Emotional_Well-being', 'MHCSF-Social_Well-being', 'MHCSF-Psychological_Well-being', 'MHCSF-Score',
                 'Honos-Somscore', 'Honos-Limitation', 'Honos-Functionality', 'Honos-Behaviour_problem', 'Honos-Symptomalogy', 'Honos-Social_Problems']
    cols_to_consider = ['Main-Age', 'Main-Biosex', 'Main-Education', 'ED_Codes', 'EDEQ-Score'] + cols_edeq + cols_dectools + cols_lav + cols_sq48 + cols_mhcsf + cols_honos
    return dict(zip(cols_to_consider, col_names))


def cols_to_float() -> dict:
    """Return a dictionary mapping column names to float types based on their substrings."""
    col_rename_dict = col_renaming()
    substrings = ['Main-Age', 'SQ48', 'MHCSF', 'Lav', 'DT', 'EDEQ', 'Honos']
    cols_renamed = {sub: [i for i in col_rename_dict.values() if sub in i] for sub in substrings}

    dtypes_to_float = {
        'DT': dict(zip(cols_renamed['DT'], FLOAT_DTYPES[:8])),
        'EDEQ': dict(zip(cols_renamed['EDEQ'], FLOAT_DTYPES[:4])),
        'Honos': dict(zip(cols_renamed['Honos'], FLOAT_DTYPES[:6])),
        'Lav': dict(zip(cols_renamed['Lav'], FLOAT_DTYPES[:4])),
        'SQ48': dict(zip(cols_renamed['SQ48'], FLOAT_DTYPES[:10])),
        'MHCSF': dict(zip(cols_renamed['MHCSF'], FLOAT_DTYPES[:4])),
    }
    return dtypes_to_float


def cols_type_cast(df: pd.DataFrame) -> (pd.DataFrame, list, dict):
    """Cast columns to appropriate data types and return the modified DataFrame along with considered columns and subscales."""
    dtypes_to_float = cols_to_float()
    col_rename_dict = col_renaming()
    df.rename(columns=col_rename_dict, inplace=True)

    main_cols = ['Main-Bsex', 'Main-Highest_Edu', 'EDtype']
    main_col_rename = ['Main-Biosex', 'Main-Education', 'ED_Codes']
    for idx, col in enumerate(main_cols):
        if col in df.columns:
            df[col] = df[col].astype('category')
            df[main_col_rename[idx]] = df[col].cat.codes.astype(float)

    cols_to_consider = ['Main-Age', 'Main-Biosex', 'Main-Education', 'ED_Codes']
    subscales = {}

    for key, dtype_dict in dtypes_to_float.items():
        res = [i for i in df.columns if key in i]
        if key != 'EDEQ':
            if res:
                for colname, dtype in dtype_dict.items():
                    df[colname] = df[colname].astype(dtype)
                cols_to_consider += list(dtype_dict.keys())
                subscales[key] = list(dtype_dict.keys())
        else:
            if len(res) > 1:
                for colname, dtype in dtype_dict.items():
                    df[colname] = df[colname].astype(dtype)
                cols_to_consider += list(dtype_dict.keys())
                subscales[key] = list(dtype_dict.keys())
            else:
                df['EDEQ-Score'] = df['EDEQ-Score'].astype(float)
                cols_to_consider.append('EDEQ-Score')
                subscales[key] = ['EDEQ-Score']

    return df[['intid', 'Split', 'EDtype'] + cols_to_consider], cols_to_consider, subscales
