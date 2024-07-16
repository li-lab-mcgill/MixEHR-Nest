# MixEHR-Nest
MixEHR-Nest: Identifying Subphenotypes within Electronic Health Records through Hierarchical Guided-Topic Modeling
## Overview
<img title="Model Overview" alt="Alt text" src="/figures/fig_model.png">

### (a) Initialization
For each patient `d`, MixEHR-Nest initializes the Phenotype topic prior for subtopic `m` within Phenotype topic `k` (`α_{d,k_m}`) by aligning the patient's ICD9-Code with Phenotype topics using an ICD-Phecode mapping. 
- **Seed Topics**: The topic prior for subtopics associated with seed topics is set to `0.9`.
- **Regular Subtopics**: The topic prior for regular subtopics is stochastically sampled from a range of `0.001` to `0.01`.

### (b) Integration and Feature Generation
MixEHR-Nest integrates information from multiple modalities within EHR data to generate multi-view EHR features. These features are then decomposed into Phenotype topic distributions for each modality `t`, as well as patient-topic mixture membership.

### (c) Probabilistic Graphical Model
In the probabilistic graphical model (PGM) representing the data generative process for patient `d` and EHR feature `j`:
- The patient-topic mixture `θ_d` is drawn from a `K×M`-dimensional Dirichlet distribution with hyperparameter `α_d`.
- The Phenotype topic `z_{d,j}` is sampled from a Categorical distribution with probabilities determined by `θ_d`.
- Given the topic assignment `z_{d,j} = k_m`, the EHR feature `X_{d,j}` is then sampled from another Categorical distribution with probabilities specified by `φ_{.,z{d, j}}`, where `φ_{.,z{d, j}}` is a `V`-dimensional Dirichlet variable with a flat hyperparameter `β`.

# Model Running Steps

This repository contains three steps to run the model using the provided Python scripts.

## Steps

### Step 0: Initialization

This step initializes the parameters for the model.

#### Usage

```bash
python step0_initialization.py --subtopic_num SUBTOPIC_NUMBER --output_dir STORE_DIR --metadata_folder METADATA_FOLDER
```

- `--subtopic_num`: Number of subtopics per phenotype (default: 3)
- `--output_dir`: Directory to store parameters (default: 'MixEHR_param')
- `--metadata_folder`: Metadata folder name (required)

### Step 1: Train Model

This step trains the model using the initialized parameters.

#### Usage

```bash
python step1_trainModel.py --subtopic_num SUBTOPIC_NUMBER --param_dir PARAM_DIR --out_dir OUT_DIR --eta ETA --iteration ITERATION
```

- `--subtopic_num`: Number of subtopics per phenotype (default: 3)
- `--param_dir`: Path to directory storing initialized parameters (default: 'MixEHR_param')
- `--out_dir`: Path to output directory (default: 'MixEHR_out')
- `--eta`: The value to downscale the contribution of non-ICD modality in training (default: 0.6)
- `--iteration`: Training iterations (default: 20)

### Step 2: Infer New Patient

This step uses the trained model to infer new patient data.

#### Usage

```bash
python step2_inferNewPat.py --subtopic_num SUBTOPIC_NUMBER --param_dir PARAM_DIR --train_out_dir TRAIN_OUT_DIR --out_dir OUT_DIR --iteration ITERATION
```

- `--subtopic_num`: Number of subtopics per phenotype (default: 3)
- `--param_dir`: Path to directory storing initialized parameters (default: 'MixEHR_param')
- `--train_out_dir`: Path to directory storing training results (default: 'MixEHR_out')
- `--out_dir`: Path to directory storing new patients infer results (default: 'MixEHR_infer')
- `--iteration`: Training iterations (default: 20)

## Example

Here is an example of how to run the steps sequentially:

```bash
# Step 0: Initialization
python step0_initialization.py --subtopic_num 3 --output_dir MixEHR_param --metadata_folder Admit

# Step 1: Train Model
python step1_trainModel.py --subtopic_num 3 --param_dir MixEHR_param --out_dir MixEHR_out --eta 0.6 --iteration 20

# Step 2: Infer New Patient
python step2_inferNewPat.py --subtopic_num 3 --param_dir MixEHR_param --train_out_dir MixEHR_out --out_dir MixEHR_infer --iteration 20
```
