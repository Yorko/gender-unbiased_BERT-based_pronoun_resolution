## Reproducing from scratch in a fresh pip environment

## 2. Instructions to reproduce the solution from scratch

**Prerequisites:** 
- Python 3.6.7
- CUDA 9.0 (in case of other CUDA versions installed, modify tensorflow version in `requirements.txt`)
- (optionally) virtualenv - to run the script in a fresh environment (otherwise, check requirements.txt to see what is going to be installed).

**Steps:** 
  1\. Create a fresh pip environment
 - `virtualenv stage1_submission_from_scratch`
 - `source stage1_submission_from_scratch/bin/activate`
 
  2\. Run preparatory script which installs all necessary dependencies (from `requirements.txt`) and downloads BERT ([uncased_L-24_H-1024_A-16](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip) and [cased_L-24_H-1024_A-16](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip)) and [ELMo](https://github.com/allenai/allennlp) models
 - `(sh preparation.sh > pip_preparation.log 2>&1 &)` 
 
  3\. The `run.py` script creates features for `gap-test` + `gap-validation` and predicts for `gap-development`. Reproduces 0.33200 Public LB loss (6th most recent team's submission).
 - `(python3 run.py > run_stage1.log 2>&1 &)`
 
  4\. Deactivate the environment
 - `deactivate`

Training logs are also provided:
 - `pip_preparation.log`
 - `run_stage1.log`
 
Running times (256 Gb RAM, Quadro P6000, 24 GiB video memory):
 - Preprocessing: 2300s (dev) and 3200s (stage_1)
 - Training: 825s (dev) and 1200s (stage_1)
