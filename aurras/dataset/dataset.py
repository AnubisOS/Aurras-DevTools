import os
import json
import random
import itertools
import glob
import pandas as pd
from pathlib import Path


class Dataset:
    """
    This class handles the loading, preprocessing, and generation of a dataset
    designed for natural language processing (NLP) tasks. It loads entities and
    intents from files, performs slot filling, creates permutations, and
    ultimately generates the final dataset.
    """

    def __init__(self, dataset_path: str, samples_per_intent: int, duplicates: bool):
        """
        Initializes the Dataset object, storing configuration parameters.

        Args:
            dataset_path: The path to the directory containing the dataset files.
            samples_per_intent: The desired number of samples per intent category.
            duplicates: A flag indicating if duplicate prompts can be used to fill the samples_per_intent quota.
        """
        self.dataset_path = dataset_path
        self.samples_per_intent = samples_per_intent
        self.duplicates = duplicates

        self.entities = {}
        self.entities_label = {}
        self.intents = {}
        self.intents_label = {}
        self.filled_prompts = {}
        self.premutated_prompts = {}
        self.generated_prompts = {}
        self.dataset = []

    def load(self):
        """
        Loads entity and intent data from the specified dataset directory.
        """
        if not os.path.isdir(self.dataset_path):
            print("Dataset directory does not exist")
            return

        entity_files = glob.glob(
            f"{self.dataset_path}/*/entities/*.entity", recursive=True
        )
        self.entities = self._load_raw(entity_files, True)
        self.entities_label = self._generate_mapping(self.entities, 1)

        intent_files = glob.glob(
            f"{self.dataset_path}/*/intents/*.intent", recursive=True
        )
        self.intents = self._load_raw(intent_files, False)
        self.intents_label = self._generate_mapping(self.intents, 0)

    def genrate_dataset(self):
        """
        Orchestrates the dataset generation process by performing slot filling,
        generating permutations, distributing entity labels, and creating the final dataset.
        """
        self._slot_filling()
        self._permutation_generation()
        self._distributed_entities_label()
        self._generate_dataset()

    def save(self, save_path=None, form: str = "csv"):
        """
        Saves the generated dataset and associated mapping files.

        Args:
            save_path: (Optional) The path where the dataset should be saved.
                       Defaults to the dataset_path specified during initialization.
            form: The desired output format for the dataset ('csv', 'pkl', or 'json').
        """
        if save_path is None:
            save_path = self.dataset_path
        
        self._save_maps(save_path)

        dataframe = pd.DataFrame(self.dataset)
        if form == "csv":
            dataframe.to_csv(save_path + "/dataset.csv")
        elif form == "pkl":
            dataframe.to_pickle(save_path + "/dataset.pkl")
        else:
            dataframe.to_json(save_path + "/dataset.json")

    def _load_raw(self, files_path: list, assign_ids: bool = False):
        """
        Loads raw data (entities or intents) from a list of file paths.

        Args:
            files_path: A list of file paths to load data from.
            assign_ids: If True, assigns unique integer IDs to each loaded item.

        Returns:
            A dictionary where keys are file stems and values are lists of loaded items.
        """
        start_id = 1
        data = {}
        for file in files_path:
            name = Path(file).stem
            samples = []
            with open(file, "r") as f:
                for line in f.readlines():
                    if not line.startswith("#"):  # Ignore comment lines
                        if assign_ids:
                            samples.append((line.lower().strip(), start_id))
                        else:
                            samples.append(line.lower().strip())
            start_id += 1
            data[name] = samples

        return data

    def _save_maps(self, save_path=None):
        """
        Saves the intent and entity label mappings to JSON files.
        """
        with open(f"{save_path}/intent_labels.json", "w") as f:
            json.dump(self.intents_label, f)

        with open(f"{save_path}/entity_labels.json", "w") as f:
            json.dump(self.entities_label, f)

    def _generate_mapping(self, data: dict, start_at: int = 0):
        """
        Creates a mapping between numerical IDs and the textual items from a dictionary.

        Args:
            data: The dictionary to generate mappings for.
            start_at: The starting value for the numerical IDs.

        Returns:
            A dictionary where keys are numerical IDs and values are the corresponding items from the input data.
        """
        counter = start_at
        labels = {}
        for item in data:
            labels[counter] = item
            counter += 1

        return labels

    def _slot_filling(self):
        """
        Replaces placeholders in intent samples with appropriate entities.
        """
        for intent in self.intents:
            self.filled_prompts[intent] = []

            for sample in self.intents[intent]:
                self.filled_prompts[intent].append(
                    [
                        (
                            self.entities[word[1:-1]] 
                            if word.startswith("{") and word.endswith("}")
                            else [(word, 0)]
                        )
                        for word in sample.split()
                    ]
                )

    def _permutation_generation(self):
        """
        Generates permutations of filled prompts to augment the dataset.
        """
        for category in self.filled_prompts:
            permutations = []

            for sample in self.filled_prompts[category]:
                permutations.extend(
                    list(
                        itertools.product(
                            *sample
                            )
                        )
                    )

            self.premutated_prompts[category] = permutations

    def _distributed_entities_label(self):
        """
        Transforms the filled prompts to include individual word entity labels.
        """
        for category in self.premutated_prompts:  # intent categories
            category_prompts = []

            for sample in self.premutated_prompts[category]:  # intent samples
                new_sample = []

                for word in sample:  # individual words / entities
                    if (
                        word[0] == ""
                    ):  # remove empty intents (mostly from prepend_request's null case)
                        continue

                    new_sample.extend(
                        [
                            (w, word[1]) 
                            for w in word[0].split()
                        ]
                    )

                category_prompts.append(new_sample)

            self.generated_prompts[category] = category_prompts

    def _generate_dataset(self):
        """
        Finalizes the dataset by selecting samples and creating the required structure.
        """
        for category in self.generated_prompts:
            # sample the category for prompts
            if (
                not self.duplicates
                and len(self.generated_prompts[category]) < self.samples_per_intent
            ):
                samples = self.generated_prompts[category]
                print(
                    f'not enough "{category}" intents were generated from templates.  Limiting number of samples to {len(self.generated_prompts[category])}'
                )
            else:
                samples = random.choices(
                    self.generated_prompts[category], k=self.samples_per_intent
                )

            for sample in range(len(samples)):
                self.dataset.append(
                    {
                        "prompts": " ".join([ w[0] for w in samples[sample] ]),
                        "prompt_intent": list(self.intents_label.values()).index(
                            category
                        ),
                        "word_entities": [w[1] for w in samples[sample]],
                    }
                )

if __name__ == '__main__': 
    
    ds = Dataset("/home/hushm/Aurras/dataset", 10, False) 
    ds.load()
    ds.genrate_dataset()
    ds.save("/home/hushm/Aurras/dataset")