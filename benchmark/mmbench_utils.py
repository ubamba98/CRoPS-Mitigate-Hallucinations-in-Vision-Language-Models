from torch.utils.data import Dataset
import math


all_options = ['A', 'B', 'C', 'D']

def is_none(value):
    if value is None:
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() == 'nan':
        return True
    if type(value) is str and value.lower() == 'none':
        return True
    return False

def get_options(row, options):
    parsed_options = []
    for option in options:
        option_value = row[option]
        if is_none(option_value):
            break
        parsed_options.append(option_value)
    return parsed_options

class MMBenchDataset(Dataset):
    def __init__(self,ds):
        self.ds = ds
    def __len__(self):
        len = 0
        for _ in self.ds:
            len += 1
        return len
    def __getitem__(self,index):
        row = self.ds[index]
        image = (((row['image'])))
        idx = row['index']
        question = row['question']
        hint = row['hint']

        temp = {"image":image,"answer":row["answer"],"index":idx,"question":question,"hint":hint}

        for options in all_options:
            temp[options] = row[options]
 
        return temp