from git import Repo
import pandas as pd
import numpy as np
import os
from fnmatch import fnmatch
import argparse
import torch
import torch.nn.functional as F

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

import sys
sys.path.insert(1, '../dataset-cleaner')
from fill_row import DesignPatternDataRow

sys.path.insert(1, '../predictor')
from predict import convert_category

## Enable cuda
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Disable wandb
import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Output file
output_file_csv = "source_output.csv" 

# Ignore files to be processed
list_excluded_folders = ['sources/bootstrap/dist', 'sources/bootstrap/js/dist']

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ignore_clones", type=bool, default=False)
    return parser.parse_args()

def classify_files_recursive(local_dir, patterns, model, tokenizer, classified_data, classifier_func):
    for entry in os.listdir(local_dir):
        full_path = os.path.join(local_dir, entry)

        exclude = False
        normalized_full_path = full_path.replace('\\', '/')
        for excluded_folder in list_excluded_folders:
            if excluded_folder in normalized_full_path:
               exclude = True 

        if os.path.isdir(full_path):
            classify_files_recursive(full_path, patterns, model, tokenizer, classified_data, classifier_func)
        elif full_path.endswith(patterns) and not exclude:
                classifier_func(classified_data, full_path, tokenizer, model)

def classify_data(data, full_path, tokenizer, model):
    try:
        with open(full_path) as f:
            print(full_path)

            code = f.read()
            tokenized_input = tokenizer(code, return_tensors='pt', padding='max_length', truncation=True, max_length=1024).to(device)
            # change prediction
            model.eval()
            with torch.no_grad():
                outputs=model(**tokenized_input)

                logits = outputs.logits
                print("Logits:", logits)

                # Convert logits to probabilities using softmax
                probabilities = F.softmax(logits, dim=-1)
                print("Probabilities:", probabilities)

                # Determine the predicted class
                predicted_class = torch.argmax(probabilities, dim=-1)
                category = convert_category(predicted_class.item())
                print("Predicted Class:", category)

                row = DesignPatternDataRow(full_path, category, '')
                with open(output_file_csv, 'a') as output:
                    output.write(f"{full_path};{category}\n")
                data.append(row)
    except:
        with open(output_file_csv, 'a') as output:
            output.write(f"{full_path};unknown;error\n")


def main():
    args = get_args()
    repos_source = pd.read_csv('source_repos.csv', header='infer', delimiter=';')
    local_dir = 'sources/'
    patterns = (".py",".cs",".java",".go",".ruby",".cpp",".c",".js",".ts",".kt",".ex",".exs")

    # clone repos
    if args.ignore_clones == True:
        for row in repos_source.itertuples():
            if row.ignore == 0:
                try:
                    print('Cloning repository ' + row.url)
                    repo = Repo.clone_from(row.url, local_dir + row.folder)
                    print('Repository cloned')
                except:
                    print('Repository already cloned')

    # Loading Model
    base_model = 'microsoft/unixcoder-base-nine'
    model_path = '../trainer/design-patterns-model'
    print("Loading Model...")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    print(device)
    print("Model loaded...")
    
    new_classified_data = []
    classify_files_recursive(local_dir, patterns, model, tokenizer, new_classified_data, classify_data)
    ## join all data and transform to a dataframe
    data = {'context': [o._content for o in new_classified_data], 'language': [o._language.value for o in new_classified_data], 
        'category': [o._category for o in new_classified_data], 'reference': [o._reference for o in new_classified_data], 
        'content_path': [o._content_path for o in new_classified_data] }

    df = pd.DataFrame(data)

    df.to_parquet('crawled_data.parquet')
    

if __name__ == "__main__":
    main()