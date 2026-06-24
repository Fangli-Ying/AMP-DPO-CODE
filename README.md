

## Prerequisites

Before running the scripts, ensure that you have the necessary dependencies installed. You can install them using:

```bash
pip install -r requirements.txt
```

Other required models and dependencies can be obtained from the following sources:

[ProtGPT2](https://huggingface.co/nferruz/ProtGPT2/tree/main),
[AMPSorter](https://drive.google.com/drive/folders/19cOtRtZzU3JAglaRFLbc5M1aMmjYTUgV?usp=drive_link),
[BioToxiPept](https://drive.google.com/drive/folders/19cOtRtZzU3JAglaRFLbc5M1aMmjYTUgV?usp=drive_link)

## Run
Here, we show how to run MulAMP to generate antimicrobial peptide sequence

Finetune protein language models.

```bash
python prefix_tuning_prot.py --batch_size 16 --epochs 50 --dataset_path ./dataset/function/amp.tsv --dataset_name function_0 --output_path ./candidate_prefix_tuning_model/
```

### 2. Candidate Antimicrobial Sequences Generation

This script generates candidate sequences using the chosen prefix-tuning model.

```bash
python generate_candidate_sequence.py --model_path ./prefix_tuning_model/
```

### 3. Candidate Sequences Evaluation
We first employ AMPSorter to predict antimicrobial activity.
```bash
python run_amp_predictor.py
```
Then we utilize BioToxiPept to predict cytotoxicity.
```bash
python run_biotoxipept.py
```

### 4. Construct preference optimization dataset and train the model:

Here we provide the preprocessed files in /dpo_dataset_amp_toxin

We then train the model on the constructed preference optimization dataset.
```bash
python train_mlpo.py --batch_size 16 --epochs 50 --lr 5e-5 --dataset_path ./dpo_dataset_amp_toxin --dataset_name function_0 --model_path ./prefix_tuning_model/function_0/
```

### 5.Generation and evaluation

Then we can generate and test the result.
```bash
python easy_inference.py
```





















