import pandas as pd
import numpy as np
import scipy
import pickle
import random
import os
from tqdm import tqdm
import argparse

class MixEHR_Nest_Test(object):
    def __init__(self, alpha, beta,
                 icd_phecode_map, Modality, folder_path, iteration =20,
                 num_sub_topics=1):
        self.alpha = alpha
        self.beta = beta
        self.icd_phecode_map = icd_phecode_map
        self.iteration = iteration
        ##T
        self.modality = Modality
        self.T = len(Modality)

        self.D = self.alpha.shape[0]
        self.K = self.alpha.shape[1] 
        self.V = list(self.modality.values())
        self.M = num_sub_topics # MINIMUM 1 SUBTOPICS, WHICH IS EQUAL TO NO SUBTOPICS
        
        self.documents = {}
        for t, modal in enumerate(self.modality):
            with open(folder_path+'/init_doc_' + str(t) + '.pkl', 'rb') as f:
                doc_t = pickle.load(f)  
            self.documents[modal] = doc_t
            self.topic_assignment = {}
            
        with open(folder_path+'/n_dw_full.pkl', 'rb') as f:
            self.document_combined = pickle.load(f)
        
        
        self.document_topic_count = np.zeros([self.D, self.K])
        self.theta = np.zeros([self.D, self.K])
        self.patient_phecode_array = {}
    
    def initialization(self):
        np.random.seed(1234)
        random.seed(1234)
        ## initialization
        for t, modal in enumerate(self.modality):
            for doc_idx, doc in enumerate(self.documents[modal]):
                words = self.documents[modal][doc]
                if t == 0:
                    #match the word to phecode idx
                    self.patient_phecode_array[doc_idx] = self.icd_phecode_map[words]
                    ## using alpha as prob to initialize the topics
                    random_topics = [np.random.randint(phecode_idx*self.M, phecode_idx*self.M + self.M, size=1)[0] for phecode_idx in self.patient_phecode_array[doc_idx]]
                    
                    if doc not in self.topic_assignment:
                        self.topic_assignment[doc_idx] = []
                    self.topic_assignment[doc_idx].extend(random_topics)
                    for word_idx, word in enumerate(words):
                        random_topic = random_topics[word_idx]
                        
                        self.document_topic_count[doc_idx,random_topic] += 1 

                else:
                    phecode_sublist = []
                    if (len(self.patient_phecode_array[doc_idx]) == 0) or (doc_idx not in self.patient_phecode_array[doc_idx]):
                        self.patient_phecode_array[doc_idx] = list(set(self.icd_phecode_map))
                    for phecode_idx in list(set(self.patient_phecode_array[doc_idx])):
                        for i in range(self.M):
                            phecode_sublist.append(phecode_idx*self.M + i)
                    random_topics = [random.choice(phecode_sublist) for word in words]
                    if doc not in self.topic_assignment:
                        self.topic_assignment[doc_idx] = []
                    self.topic_assignment[doc_idx].extend(random_topics)
                    for word_idx, word in enumerate(words):
                        random_topic = random_topics[word_idx]
                        
                        self.document_topic_count[doc_idx,random_topic] += 1
    def Gibbs_sampling_perplexity(self, phi):
        with tqdm(total = self.iteration) as pbar:
            for itr in range(self.iteration):
                for doc_idx, doc in enumerate(self.document_combined):
                    for word_idx, word in enumerate(self.document_combined[doc_idx]):
                        z = self.topic_assignment[doc_idx][word_idx]   
                        self.document_topic_count[doc_idx, z] -= 1
                        gamma_id = (self.alpha[doc_idx] + self.document_topic_count[doc_idx]) * phi[:,word]
                        prob_zid = gamma_id / np.sum(gamma_id)
                        new_z = np.random.choice(self.K, p=prob_zid)

                        ### precision too high
                        self.topic_assignment[doc_idx][word_idx] = new_z
                        self.document_topic_count[doc_idx, new_z] += 1

                ## normalization perp_theta
                for d in range(self.D):
                    self.theta[d] = (self.alpha[d] + self.document_topic_count[d]) / (np.sum(self.document_topic_count[d]) + np.sum(self.alpha[d]))
                pbar.update(1)
        return self.theta

def load_phi(training_out_dir):
    phi_ICD_path = f"{training_out_dir}/ICD/phi.pkl"
    phi_MED_path = f"{training_out_dir}/MED/phi.pkl"
    phi_CPT_path = f"{training_out_dir}/CPT/phi.pkl"
    phi_DRG_path = f"{training_out_dir}/DRG/phi.pkl"
    phi_LAB_path = f"{training_out_dir}/LAB/phi.pkl"
    phi_NOTE_path = f"{training_out_dir}/NOTE/phi.pkl"
    # theta_mtx_path = 'result_M3_train_same/theta_mtx_iter19_full.pkl' 
    with open(phi_ICD_path, 'rb') as f:
        phi_ICD = pickle.load(f)
    with open(phi_MED_path, 'rb') as f:
        phi_MED = pickle.load(f)
    with open(phi_CPT_path, 'rb') as f:
        phi_CPT = pickle.load(f)
    with open(phi_DRG_path, 'rb') as f:
        phi_DRG = pickle.load(f)
    with open(phi_LAB_path, 'rb') as f:
        phi_LAB = pickle.load(f)
    with open(phi_NOTE_path, 'rb') as f:
        phi_NOTE = pickle.load(f)

    phi = np.concatenate((phi_ICD, phi_MED, phi_CPT, phi_DRG, phi_LAB, phi_NOTE), axis=1)
    return phi

def main(M, iteration, param_dir, training_out_dir, out_dir):
    with open(f"{param_dir}/modality_meta.pkl", 'wb') as f:
        Modality = pickle.load(f)
    random.seed(0)
    phi = load_phi(training_out_dir)
    
    with open(f'{param_dir}/icd_phecode_map.pkl', 'rb') as f:
        icd_phecode_map = pickle.load(f)

    alpha_train = scipy.sparse.load_npz(f"{param_dir}/alpha_{M}.npz").toarray()
    train_model = MixEHR_Nest_Test(alpha_train, 0.001, icd_phecode_map, Modality, param_dir, iteration, M)
    train_model.initialization()
    theta = train_model.Gibbs_sampling_perplexity(phi,20)
    
    os.makedirs(f"{out_dir}", exist_ok = True)
    # os.makedirs(f"theta/{eta}", exist_ok = True)
    with open(f"{out_dir}/infered_theta.pkl", 'wb') as f:
        pickle.dump(theta, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer topic mixture matrix for new patients")
    parser.add_argument('--subtopic_num', type=int, default=3, help='Number of subtopics per phenotype (default: 3)')
    parser.add_argument('--param_dir', type=str, required=True, default='MixEHR_param', help='Path to directory store initialized parameters')
    parser.add_argument('--train_out_dir', type=str, required=True, default='MixEHR_out', help='Path to directory store training results')
    parser.add_argument('--out_dir', type=str, required=True, default='MixEHR_infer', help='Path to directory store new patients infer results')
    # parser.add_argument('--eta', type=float, required=True, default = 0.6, help='The eta value to be used in processing')
    parser.add_argument('--iteration', type=int, required=True, default = 20, help='Training iterations')
    
    args = parser.parse_args()
    main(args.subtopic_num, args.iteration, args.param_dir, args.train_out_dirn, args.out_dir)