from torch.utils.data import Dataset
from PIL import Image

class MMEDataset(Dataset):
    def __init__(self,ds,trans):
        self.ds = ds
        self.trans = trans
    def __len__(self):
        len = 0
        for _ in self.ds:
            len += 1
        return len
    def __getitem__(self,index):
        row = self.ds[index]
        image = self.trans(((row['image']).convert("RGB")))
        question_id = row['question_id']
        question = row['question']
        category = row['category']

        return {"image":image,"answer":row["answer"],"question_id":question_id,"question":question,"category":category}