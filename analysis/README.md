# analysis

- `correct_labels.py` - script that corrects `gap-test` labels according to the [published list](https://www.kaggle.com/c/gendered-pronoun-resolution/discussion/81331) coppied to `input/corrections.csv`
- `score.py` - script to compute metrics (logarithmic loss, accuracy and F1 score) for the 3 models (fine-tuned, frozen, blend). Usage: `python3 score.py -test test_file_name -result result_file_name`, `test_file_name` must be located in input folder, default is `gap-test.tsv`. `result_file_name` is where results are written, be default `result.csv` To run with corrected labels: `python score.py -test gap-test-corrected.tsv -result result_corrected.csv`.
 - `result.csv` - is a source for Table 2 in the paper
 - `result_corrected.csv` - is a source for Table 3 in the paper
