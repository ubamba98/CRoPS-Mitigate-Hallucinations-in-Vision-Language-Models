import os
import json
import pickle
import numpy as np
from benchmark.evaluators import chair_evaluator

class ChairBenchmarkDataset:
    def __init__(self,
                 coco_path: str,
                 coco_file: str,
                 base_image_path: str,
                 chair_test_size: int
                ):
        self.coco_path = coco_path
        self.coco_file = coco_file
        self.base_image_path = base_image_path
        self.image_id_key = "image_id"
        self.caption_key = "caption"
        self.images = self._load_coco_data()
        self.chair_test_size = chair_test_size

    def _load_coco_data(self) -> dict:
        with open(os.path.join(self.coco_path, self.coco_file), 'r') as f:
            lines = f.readlines()[0]
            coco_data = json.loads(lines)

        images = []
        for image in coco_data["images"]:
            images.append({
                "image_id": image["id"],
                "image_path": os.path.join(self.base_image_path, image["file_name"])
            })
        return images

    def dump_generations(self, results: list[dict], results_path: str):
        """
        Dump the results to a file. 
        results format:
        [{
            "image_id": int, # image id of the image
            "caption": str, # generated caption of the image
        }, ...]

        results_path: jsonl file
        """
        for result in results:
            with open(results_path, "a") as f:
                json.dump(result, f)
                f.write('\n')
    
    def get_test_dataset(self) -> list[dict]:
        """
        Get the test dataset.
        """
        test_dataset_path = os.path.join(self.coco_path, f"chair_test_dataset_{self.chair_test_size}.npy")
        if os.path.exists(test_dataset_path):
            test_dataset = np.load(test_dataset_path, allow_pickle=True)
        else:
            test_dataset = np.random.choice(self.images, size=self.chair_test_size, replace=False)
            np.save(test_dataset_path, test_dataset)
        return list(test_dataset)

    def evaluate(self, results_path: str, dump_results: bool = True):
        """
        Evaluate the results using the CHAIR evaluator.
        """
        evaluator_path = os.path.join(self.coco_path, "chair_evaluator.pkl")
        if os.path.exists(evaluator_path):
            evaluator = pickle.load(open(evaluator_path, "rb"))
        else:
            evaluator = chair_evaluator.CHAIR(self.coco_path)
            with open(evaluator_path, "wb") as f:
                pickle.dump(evaluator, f)

        metrics = evaluator.compute_chair(results_path, self.image_id_key, self.caption_key)

        print("CHAIR metrics:")
        self._print_metrics(metrics)

        if dump_results:
            with open(os.path.join(os.path.dirname(results_path), 'chair_results.jsonl'), 'w') as f:
                json.dump(metrics, f)
        return metrics
    
    @staticmethod
    def _print_metrics(metrics: dict):
        sentence_metrics = metrics['overall_metrics']
        
        for k, v in sentence_metrics.items():
            k_str = str(k).ljust(10)
            v_str = f'{v * 100:.01f}'
            print(k_str, v_str, sep=': ')
