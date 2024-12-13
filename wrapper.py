import os

class EvalWrapper:
    def __init__(self, dataset_name):
        self.name = dataset_name.lower()
        self.evaluate_map = {
            'iemocap': 'evaluation.evaluate_iemo',
            'ravdess': 'evaluation.evaluate_ravdess',
            'cremad-d': 'evaluation.evaluate_cremad',
            'tess': 'evaluation.evaluate_tess'
        }

    def set_eval(self):
        # Get the module path dynamically
        module_path = self.evaluate_map.get(self.name)
        if not module_path:
            supported_datasets = ', '.join(self.evaluate_map.keys())
            raise ValueError(f"Unsupported dataset name: {self.name}.\nSupported datasets are: {supported_datasets}")
        
        # Import the evaluate function dynamically
        evaluate = __import__(module_path, fromlist=['evaluate']).evaluate
        return self.name, evaluate