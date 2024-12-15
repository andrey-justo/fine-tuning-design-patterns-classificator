import argparse
from copy import deepcopy

import numpy as np
from datasets import ClassLabel, DatasetDict, load_dataset
from evaluate import load
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

## Enable cuda
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Disable wandb
import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def get_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()

# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])
    
def convert_category(prediction):
    import json
    labels_path = '../trainer/labels.json'
    with open(labels_path) as json_data:
        d = json.loads(json_data.read())
        for item in d:
            if d[item] == prediction:
                return item

    return 'unknown'

def main():
    args = get_args()

    base_model = 'microsoft/unixcoder-base-nine'
    model_path = '../trainer/design-patterns-model'
    print("Loading Model...")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=12).to(device)

    # tokenize input 
#     code_input = '''
#         // Copyright (c) Microsoft Corporation.
# // Licensed under the MIT license.
# //
# using Microsoft.Extensions.Configuration;
# using Microsoft.Extensions.DependencyInjection;
# using Microsoft.FeatureManagement;

# services.AddSingleton(configuration)
#         .AddFeatureManagement()
#         .AddFeatureFilter<AccountIdFilter>();

# //
# // Check if feature enabled
# //
# var accountServiceContext = new AccountServiceContext
# {
#     AccountId = account
# };

# bool enabled = await featureManager.IsEnabledAsync(FeatureName, accountServiceContext);
#     '''

    code_input = '''
    switch (input)
        {
            case "A":
                return new ImplementationA();
            case "B":
                return new ImplementationB();
            default:
                retrun new ImplementationDefault();
        }
'''

    tokenized_input = tokenizer(code_input, return_tensors='pt', padding='max_length', truncation=False, max_length=1024).to(device)
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
        print("Predicted Class:", convert_category(predicted_class.item()))

    print("Hey: ", tokenizer)
    print("Hey: ", model)

if __name__ == "__main__":
    main()