import pickle

import numpy as np
import omegaconf
from tqdm import trange
from src.utils import *
from src.EquationLearning.Transformers.GenerateTransformerData import Dataset, evaluate_and_wrap
import matplotlib.pyplot as plt


def create_pickle_from_data(block, path, idx):
    with open(os.path.join(path, str(idx) + ".pkl"), 'wb') as file:
        pickle.dump(block, file)


class SampleData:
    """Pre-train transformer model using generated equations"""

    def __init__(self):
        """
        Initialize TransformerTrainer class
        """
        # Read config yaml
        try:
            self.cfg = omegaconf.OmegaConf.load("src/EquationLearning/Transformers/config.yaml")
        except FileNotFoundError:
            self.cfg = omegaconf.OmegaConf.load("config.yaml")

        # Read all equations
        self.data_train_path = self.cfg.train_path
        self.data_val_path = self.cfg.val_path
        self.training_dataset = Dataset(self.data_train_path, self.cfg.dataset_train, mode="train")
        self.validation_dataset = Dataset(self.data_val_path, self.cfg.dataset_val, mode="val")
        self.word2id = self.training_dataset.word2id

    def sample_domain(self, Xs, Ys, equations):
        """Use a random domain (e.g., between -10 and 10, or -5 and 5, etc)"""
        dva = np.random.randint(3, 10)
        minX, maxX = -dva, dva
        X, Y = np.zeros((self.cfg.architecture.block_size, self.cfg.architecture.number_of_sets)), np.zeros(
            (self.cfg.architecture.block_size, self.cfg.architecture.number_of_sets))
        for ns in range(self.cfg.architecture.number_of_sets):
            # Select rows where the value of the first column is between minX and maxX
            selected_rows_indices = np.where((Xs[:, ns] >= minX) & (Xs[:, ns] <= maxX))[0]
            remaining = self.cfg.architecture.block_size - len(selected_rows_indices)
            # Randomly select 1000 rows from the selected rows
            if len(selected_rows_indices) > self.cfg.architecture.block_size:
                selected_rows_indices = np.random.choice(selected_rows_indices, self.cfg.architecture.block_size, replace=False)
            elif len(selected_rows_indices) < self.cfg.architecture.block_size and remaining < 200:
                try:
                    selected_rows_indices = list(selected_rows_indices)
                    selected_rows_indices += list(np.random.choice(np.array(selected_rows_indices), remaining, replace=False))
                    selected_rows_indices = np.array(selected_rows_indices)
                except ValueError:
                    ns, dva = 0, dva + 1  # If it failed, try with a larger domain
                    continue
            elif len(selected_rows_indices) < self.cfg.architecture.block_size and remaining >= 200:
                ns, dva = 0, dva + 1  # If it failed, try with a larger domain
                continue

            X[:, ns] = Xs[:, ns][selected_rows_indices]
            scaling_factor = 20 / (np.max(X[:, ns]) - np.min(X[:, ns]))
            X[:, ns] = (X[:, ns] - np.min(X[:, ns])) * scaling_factor - 10
            Y[:, ns] = Ys[:, ns][selected_rows_indices]
        # With a chance of 0.3, fix all sets to the same function
        if np.random.random(1) < 0.3:
            ns = np.random.randint(0, self.cfg.architecture.number_of_sets)
            X[:, 0:] = X[:, ns][:, np.newaxis]
            Y[:, 0:] = Y[:, ns][:, np.newaxis]
            equations = [equations[ns]] * self.cfg.architecture.number_of_sets
        return X, Y, equations

    def run(self):
        """Implement main training loop"""
        # Prepare list of indexes for shuffling

        batch = []
        count = 0
        n_batch = 0
        for step in trange(0, len(self.training_dataset)):  # Batch loop
            try:
                sampled_data = evaluate_and_wrap(self.training_dataset[step], self.cfg.dataset_train, self.word2id)
            except Exception as e:
                print("Problem in step = " + str(step) + " Exception = " + str(e))
                sampled_data = None

            if sampled_data is not None:
                # for i in range(sampled_data[0].shape[1]):
                #     plt.figure()
                #     plt.scatter(sampled_data[0][:, i], sampled_data[1][:, i])
                #     plt.xticks(fontsize=16)
                #     plt.yticks(fontsize=16)

                Xs, Ys, _ = self.sample_domain(sampled_data[0], sampled_data[1], sampled_data[-1])

                count += 1
                print(sampled_data[-2])
                batch.append(sampled_data)
                if count % 5000 == 0:
                    create_pickle_from_data(batch, "src/EquationLearning/Data/sampled_data/" + self.cfg.dataset +
                                            "/training", n_batch)  # /mnt/data0/data/H5datasets/sampled_data/
                    n_batch += 1
                    batch = []

        ########################################################################
        # Validation step
        ########################################################################
        batch = []
        count = 0
        n_batch = 0
        for step in trange(5000, len(self.validation_dataset)):
            try:
                sampled_data = evaluate_and_wrap(self.validation_dataset[step], self.cfg.dataset_train, self.word2id)
            except:
                sampled_data = None

            if sampled_data is not None:
                count += 1
                batch.append(sampled_data)
                if count % 1000 == 0:
                    len(batch)
                    create_pickle_from_data(batch, "src/EquationLearning/Data/sampled_data/validation", n_batch)
                    n_batch += 1
                    batch = []


if __name__ == '__main__':
    sampler = SampleData()
    sampler.run()
