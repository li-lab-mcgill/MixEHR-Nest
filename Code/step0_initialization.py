
import argparse
import numpy as np
import pandas as pd
import tqdm
import scipy.sparse
import pickle
import os
import random

# Set up argument parser with default values
parser = argparse.ArgumentParser(description='Process initialization parameters.')
parser.add_argument('--subtopic_num', type=int, default=3, help='Number of subtopics per phenotype (default: 3)')
parser.add_argument('--output_dir', type=str, default='MixEHR_param', help='Directory to store parameters (default: MixEHR_param)')
parser.add_argument('--metadata_folder', type=str, required=True, help='Metadata folder name')
args = parser.parse_args()

# Assign arguments to variables
SUBTOPIC_NUMBER = args.subtopic_num
STORE_DIR = args.output_dir
METADATA_FOLDER = args.metadata_folder
##DATA_SPLIT = 'Full', 'tain', 'test', 'val', etc.

ICD_REF = pd.read_csv(f'{METADATA_FOLDER}/D_ICD_DIAGNOSES.csv')
Modality_ICD = pd.read_csv(f'{METADATA_FOLDER}/DIAGNOSES_ICD.csv.gz')[["SUBJECT_ID", "ICD9_CODE"]].dropna(axis=0)
PHECODE_REF = pd.read_csv(f'{METADATA_FOLDER}/phecode_icd9_rolled.csv')[['ICD9', 'PheCode','ICD9 String', 'Phenotype']]

## Unified ICD9 Format
ICD_REF['ICD9_CODE'] = [icd if len(icd) == 3 else icd[:3] + '.' + icd[-1] if len(icd) == 4 else icd[:3] + '.' + icd[-2:] for icd in ICD_REF['ICD9_CODE'].tolist()]
Modality_ICD['ICD9_CODE'] = [icd if len(icd) == 3 else icd[:3] + '.' + icd[-1] if len(icd) == 4 else icd[:3] + '.' + icd[-2:] for icd in Modality_ICD['ICD9_CODE'].astype('string')]

## Map PheCodes based on ICD9 for each Patients
ICD_PHECODE_MAP = Modality_ICD.join(PHECODE_REF.set_index('ICD9'), on='ICD9_CODE', how='inner')

##icd
Modality_ICD = pd.read_csv(f'{METADATA_FOLDER}/ICD.csv',low_memory=False)
Modality_ICD['pheId'] = Modality_ICD['pheId']-1
#med
Modality_MED = pd.read_csv(f'{METADATA_FOLDER}/MED.csv',low_memory=False)
Modality_MED['pheId'] = Modality_MED['pheId']-1
##cpt
Modality_CPT = pd.read_csv(f'{METADATA_FOLDER}/CPT.csv',low_memory=False)
Modality_CPT['pheId'] = Modality_CPT['pheId']-1
#drg
Modality_DRG = pd.read_csv(f'{METADATA_FOLDER}/DRG.csv',low_memory=False)
Modality_DRG['pheId'] = Modality_DRG['pheId']-1
##lab
Modality_LAB = pd.read_csv(f'{METADATA_FOLDER}/Lab.csv',low_memory=False)
Modality_LAB['pheId'] = Modality_LAB['pheId']-1
#note
Modality_Note = pd.read_csv(f'{METADATA_FOLDER}/Note.csv',low_memory=False)
Modality_Note['pheId'] = Modality_Note['pheId']-1

## ICD List
ICD_meta = pd.read_csv(f'{METADATA_FOLDER}/ICD_wordId.csv')
ICD_meta['pheId'] = ICD_meta['pheId']-1
ICD_meta['ICD9_CODE'] = [icd if len(icd) == 3 else icd[:3] + '.' + icd[-1] if len(icd) == 4 else icd[:3] + '.' + icd[-2:] for icd in ICD_meta['ICD9_CODE'].tolist()]
ICD_PHECODE_MAP_tmp = ICD_PHECODE_MAP[['ICD9_CODE','PheCode']]
ICD_meta = ICD_meta.merge(ICD_PHECODE_MAP_tmp, on = 'ICD9_CODE', how = 'left').drop_duplicates()
ICD_list = ICD_meta.ICD9_CODE.tolist()
## MED List
MED_meta = pd.read_csv(f'{METADATA_FOLDER}/MED_wordId.csv')
MED_meta['pheId'] = MED_meta['pheId']-1
MED_list = MED_meta.DRUG_KEY.tolist()
## CPT List
CPT_meta = pd.read_csv(f'{METADATA_FOLDER}/CPT_wordId.csv')
CPT_meta['pheId'] = CPT_meta['pheId']-1
CPT_list = CPT_meta.ICD9_CODE.tolist()
## DRG List
DRG_meta = pd.read_csv(f'{METADATA_FOLDER}/DRG_wordId.csv')
DRG_meta['pheId'] = DRG_meta['pheId']-1
DRG_list = DRG_meta.DRG_KEY.tolist()
## LAB List
LAB_meta = pd.read_csv(f'{METADATA_FOLDER}/Lab_wordId.csv')
LAB_meta['pheId'] = LAB_meta['pheId']-1
LAB_list = LAB_meta.ITEMID.tolist()
## Note List
Note_meta = pd.read_csv(f'{METADATA_FOLDER}/Note_wordId.csv')
Note_meta['pheId'] = Note_meta['pheId']-1
Note_list = Note_meta.term.tolist()

V = len(ICD_list) + len(MED_list) + len(CPT_list) + len(DRG_list) + len(LAB_list) + len(Note_list)
## PheCode List
PheCode_List = np.sort(ICD_PHECODE_MAP.PheCode.unique().tolist())
K = len(PheCode_List)
## Patient List
patient_list = np.sort(Modality_ICD.SUBJECT_ID.unique().tolist() + Modality_MED.SUBJECT_ID.unique().tolist()
+ Modality_CPT.SUBJECT_ID.unique().tolist() + Modality_DRG.SUBJECT_ID.unique().tolist() 
+ Modality_LAB.SUBJECT_ID.unique().tolist() + Modality_Note.SUBJECT_ID.unique().tolist())
patient_list = list(set(patient_list))
D = len(patient_list)
# ###skip this if you're using all patients
# from sklearn.model_selection import train_test_split
# import random
# random.seed(0)
# random_state = 21
# ### split train/test patients
# pat_train, pat_test = train_test_split(patient_list, test_size=0.5, random_state=random_state)
# pat_test, pat_val = train_test_split(pat_test, test_size=0.5, random_state=random_state)
# D = len(pat_test)
# patient_list = pat_test
# data_split = 'test'
# if not os.path.isdir(f"{data_split}_{random_state}"):
#     os.mkdir(f"{data_split}_{random_state}")
Modality = {'ICD': len(ICD_list), 'MED': len(MED_list), 'CPT': len(CPT_list), 'DRG': len(DRG_list), 'LAB': len(LAB_list), 'NOTE': len(Note_list)}

for key, value in Modality.items():
    print(f"Modality {key} has {value} features.")
# print( , len(MED_list) , len(CPT_list) , len(DRG_list) , len(LAB_list), len(Note_list))
with open(f"{STORE_DIR}/modality_meta.pkl", 'wb') as f:
    pickle.dump(Modality, f)
    
print(f"# of subphenotype = {K}")
print(f"# of patients = {D}")

document_combined = {}
for i in range(D):
    document_combined[i] = []

init_doc = {}
icd_subject = Modality_ICD.SUBJECT_ID.unique().tolist()
with tqdm.tqdm(total = len(patient_list)) as pbar:
    for i,subject_id in enumerate(patient_list):
        tmp = []
        if subject_id in icd_subject:
            icd_tmp = Modality_ICD[Modality_ICD.SUBJECT_ID == subject_id]
            for index, row in icd_tmp.iterrows():
                word_freq = int(row["freq"])
                j = int(row["pheId"])
                while(word_freq!=0):
                    tmp.append(j)
                    document_combined[i].append(j)
                    word_freq -=1
        init_doc[i] = tmp

        random.shuffle(init_doc[i])
        pbar.update(1)
with open(f"{STORE_DIR}/init_doc_0.pkl", 'wb') as f:
    pickle.dump(init_doc, f)        

init_doc = {}
med_subject = Modality_MED.SUBJECT_ID.unique().tolist()
with tqdm.tqdm(total = len(patient_list)) as pbar:
    for i,subject_id in enumerate(patient_list):
        tmp = []
        if subject_id in med_subject:
            med_tmp = Modality_MED[Modality_MED.SUBJECT_ID == subject_id]
            for index, row in med_tmp.iterrows():
                word_freq = int(row["freq"])
                j = int(row["pheId"])
                while(word_freq!=0):
                    tmp.append(j)
                    document_combined[i].append(j+len(ICD_list))
                    word_freq -=1
        init_doc[i] = tmp
        random.shuffle(init_doc[i])
        pbar.update(1)
with open(f"{STORE_DIR}/init_doc_1.pkl", 'wb') as f:
    pickle.dump(init_doc, f)        

init_doc = {}
cpt_subject = Modality_CPT.SUBJECT_ID.unique().tolist()
with tqdm.tqdm(total = len(patient_list)) as pbar:
    for i,subject_id in enumerate(patient_list):
        tmp = []
        if subject_id in cpt_subject:
            cpt_tmp = Modality_CPT[Modality_CPT.SUBJECT_ID == subject_id]
            for index, row in cpt_tmp.iterrows():
                word_freq = int(row["freq"])
                j = int(row["pheId"])
                while(word_freq!=0):
                    tmp.append(j)
                    document_combined[i].append(j+len(ICD_list)+len(MED_list))
                    word_freq -=1
        init_doc[i] = tmp
        random.shuffle(init_doc[i])
        pbar.update(1)
with open(f"{data_split}_{random_state}/init_doc_2.pkl", 'wb') as f:
    pickle.dump(init_doc, f)        

init_doc = {}
drg_subject = Modality_DRG.SUBJECT_ID.unique().tolist()
with tqdm.tqdm(total = len(patient_list)) as pbar:
    for i,subject_id in enumerate(patient_list):
        tmp = []
        if subject_id in drg_subject:
            drg_tmp = Modality_DRG[Modality_DRG.SUBJECT_ID == subject_id]
            for index, row in drg_tmp.iterrows():
                word_freq = int(row["freq"])
                j = int(row["pheId"])
                while(word_freq!=0):
                    tmp.append(j)
                    document_combined[i].append(j+len(ICD_list)+len(MED_list)+len(CPT_list))
                    word_freq -=1
        init_doc[i] = tmp
        random.shuffle(init_doc[i])
        pbar.update(1)
with open(f"{STORE_DIR}/init_doc_3.pkl", 'wb') as f:
    pickle.dump(init_doc, f)        

init_doc = {}
lab_subject = Modality_LAB.SUBJECT_ID.unique().tolist()
with tqdm.tqdm(total = len(patient_list)) as pbar:
    for i,subject_id in enumerate(patient_list):
        tmp = []
        if subject_id in lab_subject:
            lab_tmp = Modality_LAB[Modality_LAB.SUBJECT_ID == subject_id]
            for index, row in lab_tmp.iterrows():
                word_freq = int(row["freq"])
                j = int(row["pheId"])
                while(word_freq!=0):
                    tmp.append(j)
                    document_combined[i].append(j+len(ICD_list)+len(MED_list)+len(CPT_list)+len(DRG_list))

                    word_freq -=1
        init_doc[i] = tmp
        random.shuffle(init_doc[i])
        pbar.update(1)
with open(f"{STORE_DIR}/init_doc_4.pkl", 'wb') as f:
    pickle.dump(init_doc, f)        

init_doc = {}
note_subject = Modality_Note.SUBJECT_ID.unique().tolist()
with tqdm.tqdm(total = len(patient_list)) as pbar:
    for i,subject_id in enumerate(patient_list):
        tmp = []
        if subject_id in note_subject:
            note_tmp = Modality_Note[Modality_Note.SUBJECT_ID == subject_id]
            for index, row in note_tmp.iterrows():
                word_freq = int(row["freq"])
                j = int(row["pheId"])
                while(word_freq!=0):
                    tmp.append(j)
                    document_combined[i].append(j+len(ICD_list)+len(MED_list)+len(CPT_list)+len(DRG_list)+len(LAB_list))
                    word_freq -=1
        init_doc[i] = tmp
        random.shuffle(init_doc[i])
        pbar.update(1)
with open(f"{STORE_DIR}/init_doc_5.pkl", 'wb') as f:
    pickle.dump(init_doc, f)        

with open(f"{STORE_DIR}/n_dw_full.pkl", 'wb') as f:
    pickle.dump(document_combined, f)  

idx_patient_meta = {v: k for v, k in enumerate(patient_list)}
with open(f"{STORE_DIR}/idx_patient_map.pkl", 'wb') as f:
    pickle.dump(idx_patient_meta, f)

# alpha (K * M)
print('Subtopic number is', SUBTOPIC_NUMBER)
M = SUBTOPIC_NUMBER
ICD_Phecode_MAP_tmp = Modality_ICD.merge(ICD_meta, on = 'pheId', how = 'left')
ICD_Phecode_MAP_tmp = ICD_Phecode_MAP_tmp[['SUBJECT_ID',"PheCode", 'freq']].groupby(['SUBJECT_ID',"PheCode"]).sum().reset_index()
# Initalize alpha .
alpha_dk = np.random.uniform(low=0.001, high=0.01, size=(D, K, M))
flatten_alpha_dk = []

print('Initialize alpha...')
icd_subject = Modality_ICD.SUBJECT_ID.unique().tolist()
with tqdm.tqdm(total = D) as pbar:
    for idx, pat in enumerate(patient_list):
        if subject_id in icd_subject:
            # get phecode idx for patient 
            phecode = ICD_Phecode_MAP_tmp[ICD_Phecode_MAP_tmp.SUBJECT_ID == pat].PheCode.tolist()
            d_k_idx = [list(PheCode_List).index(phe) for phe in phecode]
            # assign 1 to the subtopics of phecodes
            alpha_dk[idx][d_k_idx] = 0.9     
        # flatten alpha_dk for patient d
        flatten_alpha_dk.append(alpha_dk[idx].flatten())
        pbar.update(1)

flatten_alpha_dk = np.array(flatten_alpha_dk)

## Save generated alpha
sparse_alpha = scipy.sparse.csc_matrix(flatten_alpha_dk)
scipy.sparse.save_npz(f"{STORE_DIR}/alpha_{M}.npz", sparse_alpha) 
print('Alpha with ' + str(M) + ' Subtopics Saved.')