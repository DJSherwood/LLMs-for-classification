import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import urllib.request
from DataTransformation import DataTransformation
import polars as pl

class EditorDataset(Dataset):
    def __init__(self, tokens_from_editors, max_length=35, pad_token_id=22704):
        self.encoded_texts = tokens_from_editors
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __get__item__(self, index):
        encoded = self.encoded_texts[index]
        label = self.

if __name__ is "__main__":
    passages = {
        "D": ["Deut 6", "Deut 12-13", "Deut 15-16", "Deut 18-19", "Deut 26", "Deut 28"],
        "DH": ["Deut 8-11", "Deut 27", "Josh 1", "Josh 5", "Josh 6", "Josh 12", "Josh 23",
               "Judg 2", "Judg 6", "2Sam 7", "1Kgs 8", "2Kgs 17:1-21", "2Kgs 22-25"],
        "P": ["Gen 1:1-31", "Gen 2:1-3", "Gen 5:3-28", "Gen 5:30-32", "Gen 6:9-22", "Gen 9:1-17",
              "Gen 6:28-29", "Gen 10:2-7", "Gen 10:20", "Gen 10:22-23", "Gen 10:31", "Gen 11:11-26",
              "Gen 11:29-32", "Gen 12:5", "Gen 13:6", "Gen 13:12", "Gen 16:3", "Gen 16:15-16", "Gen 21:2-5",
              "Gen 22:20-24", "Gen 23:1-20", "Gen 25:7-10", "Gen 25:13-17", "Gen 25:20", "Gen 26:20",
              "Gen 26:34-35", "Gen 27:46", "Gen 28:1-9", "Gen 35:9-15", "Gen 35:27-29", "Gen 36:40-43",
              "Gen 37:1", "Gen 46:6-7", "Gen 47:28", "Gen 49:29-33", "Gen 50:12-13", "Exod 1:1-4", "Exod 1:7",
              "Exod 1:13-14", "Exod 2:23-25", "Exod 7:1-13", "Exod 7:19-22", "Exod 8:1-3", "Exod 8:11-15",
              "Exod 9:8-12", "Exod 11:9-10", "Exod 12:40-42", "Exod 13:20", "Exod 14:1-4", "Exod 14:8-10",
              "Exod 14:15-18", "Exod 14:21-23", "Exod 14:27-29", "Exod 15:22", "Exod 19:1", "Exod 24:16-17",
              "Gen 17", "Exod 6", "Exod 16", "Exod 25-31", "Exod 35-40", "Lev 1-4", "Exod 8-9"]
    }
    # Step through the procedure
    dt = DataTransformation(file_name='wlc.txt', passages=passages)
    dt.initial_transform()
    dt.assign_editors()
    dt.add_training_testing()
    flattened_tokens = dt.convert_to_torch(column_name='Token')

    # Create datasets
