from torch.utils.data import Dataset
from PIL import Image

class MMEDataset(Dataset):
    def __init__(self,ds):
        self.ds = ds
        # self.trans = trans
    def __len__(self):
        len = 0
        for _ in self.ds:
            len += 1
        return len
    def __getitem__(self,index):
        row = self.ds[index]
        # image = self.trans(((row['image']).convert("RGB")))
        image = (row['image']).convert("RGB")
        question_id = row['question_id']
        question = row['question']
        category = row['category']

        return {"image":image,"answer":row["answer"],"question_id":question_id,"question":question,"category":category}

# Predefined category classification
eval_type_dict = {
    "Perception": [
        "existence", "count", "position", "color", "posters", 
        "celebrity", "scene", "landmark", "artwork", "OCR"
    ],
    "Cognition": [
        "commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"
    ]
}

# Function to parse model output into "yes" or "no"
def parse_pred_ans(pred_ans):
    pred_ans = pred_ans.lower().strip().replace(".", "")
    if pred_ans in ["yes", "no"]:
        return pred_ans
    elif pred_ans.startswith("y"):
        return "yes"
    elif pred_ans.startswith("n"):
        return "no"
    return "other"

