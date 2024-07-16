import pandas as pd
import numpy as np
import scipy
import pickle
import random
import os
from tqdm import tqdm
import argparse

class MixEHR_Nest(object):
    def __init__(self, alpha, beta, icd_phecode_map, Modality, 
                 downscale_weight, param_dir, out_dir, iteration = 20, num_sub_topics=1):
        self.alpha = alpha
        self.beta = beta
        self.downscale_weight = downscale_weight
        self.iteration = iteration
        self.out_dir = out_dir
        self.param_dir = param_dir
        
        self.icd_phecode_map = icd_phecode_map
        
        ##T
        self.modality = Modality
        self.T = len(Modality)
        
        self.D = self.alpha.shape[0]
        self.K = self.alpha.shape[1] 
        self.V = list(self.modality.values())

        self.M = num_sub_topics # MINIMUM 1 SUBTOPICS, WHICH IS EQUAL TO NO SUBTOPICS
        
        
        self.documents = {}
        self.topic_assignment = {}
        self.topic_word_count = {}
        self.topic_count = {}
        self.phi = {}
        for t, modal in enumerate(self.modality):
            with open(f'{self.param_dir}/init_doc_' + str(t) + '.pkl', 'rb') as f:
                doc_t = pickle.load(f)   
            self.documents[modal] = doc_t
            self.topic_assignment[modal] = {}
            self.topic_word_count[modal] = np.zeros([self.K, self.V[t]])
            self.phi[modal] = np.zeros([self.K, self.V[t]])
            self.topic_count[modal] = np.zeros(self.K)
    
        
        
        ##n_dk
        self.document_topic_count = np.zeros([self.D, self.K])
        self.theta = np.zeros([self.D, self.K])
        self.theta_ICD = np.zeros([self.D, self.K])
        
        ## Create result directory
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        for key, value in self.modality.items():   
            if not os.path.isdir(os.path.join(out_dir, str(key))):
                os.mkdir(os.path.join(out_dir, str(key)))
        self.patient_phecode_array = {}
        
        print("# of DOCS:", self.D)                            
        print("# of TOPICS:", self.K//self.M)
        print("# of MODALITIES:", self.T)
        print("# of SUBTOPICS:", self.M)
        print("MODALITIES / # of VOCAB:", self.modality)
        
        print("Downscale for other modality by:", self.downscale_weight)
        print("Training iterations:", self.iteration)

    def Training(self):
        print("Initializing for ICD modality...")
        self.initialize_ICD()
        print("Sampling for ICD modality...")
        self.Gibbs_sampling_ICD()
        
        print("Initializing for other modality...")
        self.initialize_other()
        print("Sampling for other modality...")
        self.Gibbs_sampling_other()
        
    def save_phi(self, modal, i):
        phi_mtx_path = os.path.join(self.out_dir,str(modal),'phi_iter' + str(i) + '.pkl')
        with open(phi_mtx_path, 'wb') as f:
            pickle.dump(self.phi[modal], f)
        if i == self.iteration-1 :
            phi_mtx_path = os.path.join(self.out_dir, str(modal), 'phi.pkl')
            with open(phi_mtx_path, 'wb') as f:
                pickle.dump(self.phi[modal], f)
    
    def save_theta(self, modal, i):
        if modal == 'ICD':
            theta_mtx_path = os.path.join(self.out_dir,'theta_ICD_mtx_iter' + str(i) + '.pkl') 
            with open(theta_mtx_path, 'wb') as f:
                pickle.dump(self.theta_ICD, f)
        else:
            theta_mtx_path = os.path.join(self.out_dir,'theta_mtx_iter' + str(i) + '.pkl') 
            with open(theta_mtx_path, 'wb') as f:
                pickle.dump(self.theta, f)
            if i == self.iteration-1 :
                theta_mtx_path = os.path.join(self.out_dir, 'theta.pkl')
                with open(theta_mtx_path, 'wb') as f:
                    pickle.dump(self.theta, f)

    def initialize_ICD(self):
        modal = 'ICD'
        np.random.seed(1234)
        random.seed(1234)
        for doc_idx, doc in enumerate(self.documents[modal]):
            words = self.documents[modal][doc]
            #match the word to phecode idx
            self.patient_phecode_array[doc_idx] = self.icd_phecode_map[words]
            ## using alpha as prob to initialize the topics
            random_topics = [np.random.randint(phecode_idx*self.M, phecode_idx*self.M + self.M, size=1)[0] for phecode_idx in self.patient_phecode_array[doc_idx]]

            self.topic_assignment[modal][doc_idx] = random_topics
            for word_idx, word in enumerate(words):
                random_topic = random_topics[word_idx]
                ##
                self.topic_word_count[modal][random_topic,word] += 1
                ## n_dk share across all modality, exclus from medication
                self.document_topic_count[doc_idx,random_topic] += 1 
                self.topic_count[modal][random_topic] += 1
    
    def initialize_other(self):
        np.random.seed(1234)
        random.seed(1234)
        for t, modal in enumerate(self.modality):
            for doc_idx, doc in enumerate(self.documents[modal]):
                words = self.documents[modal][doc]
                if t == 0:
                    continue
                else:
                    phecode_sublist = []
                    if (len(self.patient_phecode_array[doc_idx]) == 0) or (doc_idx not in self.patient_phecode_array[doc_idx]):
                        self.patient_phecode_array[doc_idx] = list(set(self.icd_phecode_map))
                    for phecode_idx in list(set(self.patient_phecode_array[doc_idx])):
                        for i in range(self.M):
                            phecode_sublist.append(phecode_idx*self.M + i)
                    random_topics = [random.choice(phecode_sublist) for word in words]
                    self.topic_assignment[modal][doc_idx] = random_topics
                    for word_idx, word in enumerate(words):
                        random_topic = random_topics[word_idx]

                        self.document_topic_count[doc_idx,random_topic] += 1 * self.downscale_weight 
                        self.topic_word_count[modal][random_topic,word] += 1 
                        self.topic_count[modal][random_topic] += 1
    
    def Gibbs_sampling_ICD(self):
        np.random.seed(1234)
        random.seed(1234)
        modal = 'ICD'
        t = 0
        with tqdm.tqdm(total = self.iteration) as pbar:
            for i in range(self.iteration):
                for doc_idx, doc in enumerate(self.documents[modal]):
                    for word_idx, word in enumerate(self.documents[modal][doc]):

                        z = self.topic_assignment[modal][doc_idx][word_idx]              
                        self.topic_word_count[modal][z,word] -= 1
                        self.document_topic_count[doc_idx, z] -= 1
                        self.topic_count[modal][z] -= 1

                        ##Sampling...
                        gamma_id = (self.alpha[doc_idx] + self.document_topic_count[doc_idx]) * ((self.beta + self.topic_word_count[modal][:, word]) / (self.beta * self.V[t] + self.topic_count[modal]))
                        prob_zid = gamma_id / np.sum(gamma_id)
                        new_z = np.random.choice(self.K, p=prob_zid)

                        self.topic_assignment[modal][doc_idx][word_idx] = new_z
                        self.topic_word_count[modal][new_z,word] += 1
                        self.document_topic_count[doc_idx, new_z] += 1
                        self.topic_count[modal][new_z] += 1

                ## normalization phi and theta
                for k in range(self.K):
                    self.phi[modal][k] = (self.beta + self.topic_word_count[modal][k]) / (self.beta * self.V[t] + self.topic_count[modal][k])
                self.save_phi(modal, i)
                for d in range(self.D):
                    self.theta_ICD[d] = (self.alpha[d] + self.document_topic_count[d]) / (np.sum(self.document_topic_count[d]) + np.sum(self.alpha[d]))
                self.save_theta(modal, i)
                pbar.update(1)
    def Gibbs_sampling_other(self): 
        np.random.seed(1234)
        random.seed(1234)
        ## Collapsed Gibbs Sampling
        with tqdm.tqdm(total = self.iteration) as pbar:
            for i in range(self.iteration):
                for t, modal in enumerate(self.modality):
                    if t == 0:
                        continue
                    else:
                        for doc_idx, doc in enumerate(self.documents[modal]):
                            for word_idx, word in enumerate(self.documents[modal][doc]):
                                z = self.topic_assignment[modal][doc_idx][word_idx]              
                                self.topic_word_count[modal][z,word] -= 1
                                self.document_topic_count[doc_idx, z] -= 1 * self.downscale_weight
                                self.topic_count[modal][z] -= 1

                                ##Sampling...
                                gamma_id = (self.alpha[doc_idx] + self.document_topic_count[doc_idx]) * ((self.beta + self.topic_word_count[modal][:, word]) / (self.beta * self.V[t] + self.topic_count[modal]))
                                prob_zid = gamma_id / np.sum(gamma_id)
                                new_z = np.random.choice(self.K, p=prob_zid)

                                self.topic_assignment[modal][doc_idx][word_idx] = new_z
                                self.topic_word_count[modal][new_z,word] += 1 
                                self.document_topic_count[doc_idx, new_z] += 1 * self.downscale_weight
                                self.topic_count[modal][new_z] += 1

                        ## normalization phi and theta
                        for k in range(self.K):
                            self.phi[modal][k] = (self.beta + self.topic_word_count[modal][k]) / (self.beta * self.V[t] + self.topic_count[modal][k])
                        self.save_phi(modal, i)

                for d in range(self.D):
                    self.theta[d] = (self.alpha[d] + self.document_topic_count[d]) / (np.sum(self.document_topic_count[d]) + np.sum(self.alpha[d]))
                self.save_theta(modal, i)
                pbar.update(1)     
                
        print("Finish Training.")

### FUll size training

def main(M,param_dir,out_dir, eta, iteration):
    M = 3
    with open(f"{param_dir}/modality_meta.pkl", 'wb') as f:
        Modality = pickle.load(f)
    random.seed(0)

    alpha = scipy.sparse.load_npz(f"{param_dir}/alpha_{M}.npz").toarray()
    with open(f'{param_dir}/icd_phecode_map.pkl', 'rb') as f:
        icd_phecode_map = pickle.load(f)

    model = MixEHR_Nest(alpha, 0.001, icd_phecode_map, Modality, 
                 eta, param_dir, out_dir, iteration, num_sub_topics=M)
    model.Training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process patient data with given eta value")
    parser.add_argument('--subtopic_num', type=int, default=3, help='Number of subtopics per phenotype')
    parser.add_argument('--param_dir', type=str, default='MixEHR_param', help='Path to directory store initialized parameters')
    parser.add_argument('--out_dir', type=str, default='MixEHR_out', help='Path to output directory')
    parser.add_argument('--eta', type=float, default = 0.6, help='The eta value to be used in processing')
    parser.add_argument('--iteration', type=int, default = 20, help='Training iterations')
    args = parser.parse_args()
    main(args.subtopic_num, args.param_dir, args.out_dir, args.eta, args.iteration)