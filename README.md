# Jooble test task

### Prerequisites

Python 3.7 or later with all requirements.txt dependencies installed. To install run:
```
pip install -r requirements.txt 
```

## Running

Put `train.tsv` and `test.tsv` dataframes to `/data` folder and run the command:

```
python data_preprocessing.py
``` 

You can run command with parameters:

`--train_path` - path to training dataframe

`--test_path` - path to test dataframe

`--test_proc_save_path` - path to save preprocessed dataframe (save to `/data` by default)

`--chunksize` - preprocessing batch size(set to `100` by default)

## Authors

* **Timofey Tkachenko** - [Linkedin](https://www.linkedin.com/in/timofey-tkachenko-928627175)


