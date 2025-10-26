"""
Code to create a Huggingface dataset from a pandas dataframe and save it to disk

To run the script, with following command:
python hf_textprocessing.py

Following libraries need to be installed:
pip install datasets pandas scikit-learn
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, Features, Value, ClassLabel
import re

# Get the data 
def get_data(file_name, columns):

    # Read the data
    data_df = pd.read_csv(file_name,usecols=columns)

    # Rename columns to text and label
    data_df.columns = ['text', 'label']
    data_df.text = data_df.text.astype(str)
    data_df.label = data_df.label.astype(str)

    # Remove nan values
    data_df = data_df.dropna()

    return data_df

def show_data(datasets) -> None:

    sample_texts = datasets['train'].select(range(3))

    sample_texts_processed = sample_texts.map(text_preprocessing)
    # Print org and then processing one by one 
    for i in range(3):
        print("Processed")
        print(f"Org:\n {sample_texts['text'][i]}")
        print(f"Processed:\n {sample_texts_processed['text'][i]}")
        print("\n")

def text_preprocessing(text_dict):

    text = text_dict["text"]

    # TODO: Add more preprocessing steps

    return {"text": text}

if __name__ == "__main__":

    # Set the text and label columns
    seed = 42
    columns = ['text', 'label']
    file_name = 'data.csv'
    path_to_data_folder = 'data'
    labels = ['0','1']

    data_df = get_data(file_name, columns)

    # Create Huggingface dataset from pandas dataframe with train test split
    train_df, test_df = train_test_split(data_df, test_size=0.2, 
                                         random_state=seed,stratify=data_df['label'])

    # Define the schema for the dataset
    features = Features({
        'text': Value('string'),
        'label': ClassLabel(num_classes=2,names=labels)
    })

    # Combine now train and test datasets
    datasets = DatasetDict({
        "train": Dataset.from_pandas(train_df, preserve_index=False,features=features),
        "test": Dataset.from_pandas(test_df,preserve_index=False,features=features)})

    # Print the dataset
    print(datasets)

    # Show sample of the dataset
    show_data(datasets)

    # Save the dataset
    datasets.save_to_disk(path_to_data_folder)