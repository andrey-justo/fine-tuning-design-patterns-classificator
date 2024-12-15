import pandas as pd
from os import listdir
from os.path import isfile, join, abspath
from fill_row import DesignPatternDataRow
import math

dataset_file = pd.read_csv('projeto-final/design-patterns-dataset.csv', header='infer', delimiter=';')

collection_curated_data = []

# The cleaner should prepare the csv with the format
# context | language | category | reference
# context: needs to be read by a given row like context-path-folder or context-path
# language: needs to be apply based on file extension
# category: inplace or can be extracted by folder structure
# refernce: for debugging and for Copyright 
for row in dataset_file.itertuples():
    content_path = ''
    is_folder = False
    if row.context_path is not None and not math.isnan(row.context_path):
        content_path = row.context_path
    elif row.context_path_folder:
        is_folder = True
        content_path = row.context_path_folder

    if not content_path:
        # error configuration
        raise Exception("Please configure content path")

    category = row.category
    reference = row.url_reference

    ## FIXME: Improve reading files
    content_path = abspath('projeto-final/' + content_path)
    if is_folder:
        onlyfiles = [f for f in listdir(content_path) if isfile(join(content_path, f))]
        for file in onlyfiles:
            collection_curated_data.append(DesignPatternDataRow(content_path +  '\\' + file, category, reference))
    else:
        collection_curated_data.append(DesignPatternDataRow(content_path, category, reference))
    
## join all data and transform to a dataframe
data = {'context': [o._content for o in collection_curated_data], 'language': [o._language.value for o in collection_curated_data], 
        'category': [o._category for o in collection_curated_data], 'reference': [o._reference for o in collection_curated_data], 
        'content_path': [o._content_path for o in collection_curated_data] }

df = pd.DataFrame(data)

df.to_parquet('output.parquet')