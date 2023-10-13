import glob
import torch
import omegaconf
from torch import nn
from tqdm import trange
from src.utils import *
from torch import optim
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from src.EquationLearning.Transformers.model import Model
from src.EquationLearning.Transformers.GenerateTransformerData import Dataset

# import time


def open_pickle(path):
    with open(path, 'rb') as file:
        block = pickle.load(file)
    return block


def loss_sample(output, trg):
    """Loss function that combines cross-entropy and information entropy for a single sample"""
    # Cross-entropy
    ce = nn.CrossEntropyLoss(ignore_index=0)
    L1 = ce(output, trg)

    # Information entropy
    # L2 = torch.tensor(0, device=z_sets.device, dtype=z_sets.dtype)
    # for i in range(z_sets.shape[2]):  # Across the embedding dimension
    #     q = z_sets[:, :, i]
    #     p = nn.functional.softmax(q, dim=0)
    #     logp = nn.functional.log_softmax(q, dim=0)
    #     # Calculate entropy
    #     L2 += torch.sum(torch.mul(p, logp))
    return L1  # , L2 / z_sets.shape[2]


class TransformerTrainer:
    """Pre-train transformer model using generated equations"""

    def __init__(self):
        """
        Initialize TransformerTrainer class
        """
        # Read config yaml
        try:
            self.cfg = omegaconf.OmegaConf.load("src/EquationLearning/Transformers/config.yaml")
        except FileNotFoundError:
            self.cfg = omegaconf.OmegaConf.load("../Transformers/config.yaml")

        # Read all equations
        self.sampledData_train_path = 'src/EquationLearning/Data/sampled_data/' + self.cfg.dataset + '/training'
        self.sampledData_val_path = 'src/EquationLearning/Data/sampled_data/' + self.cfg.dataset + '/validation'
        self.data_train_path = self.cfg.train_path
        self.training_dataset = Dataset(self.data_train_path, self.cfg.dataset_train, mode="train")
        self.word2id = self.training_dataset.word2id

        # Load model
        self.model = Model(cfg=self.cfg.architecture, cfg_inference=self.cfg.inference, word2id=self.word2id)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Training parameters
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=self.cfg.architecture.lr)
        self.writer = SummaryWriter('runs')
        self.lambda_ = self.cfg.dataset_train.lambda_

    def fit(self):
        """Implement main training loop"""
        epochs = self.cfg.epochs
        batch_size = self.cfg.batch_size
        # Get names of training and val blocks
        train_files = glob.glob(os.path.join(self.sampledData_train_path, '*.pkl'))
        val_files = glob.glob(os.path.join(self.sampledData_val_path, '*.pkl'))
        # Prepare list of indexes for shuffling
        indexes = np.arange(len(train_files))

        # self.model.load_state_dict(torch.load('src/EquationLearning/models/saved_models/Model1'))
        print("""""""""""""""""""""""""""""")
        print("Start training")
        print("""""""""""""""""""""""""""""")
        global_batch_count = 0
        for epoch in trange(epochs):  # Epoch loop
            # Shuffle indices
            np.random.shuffle(indexes)

            batch_count = 0
            for b_ind in indexes:  # Block loop (each block contains 1000 inputs)
                # Read block
                block = open_pickle(train_files[b_ind])
                T = np.ceil(1.0 * len(block) / batch_size).astype(np.int32)

                # Format elements in the block as torch Tensors
                XY_block = torch.zeros((len(block), block[0][0].shape[0], 2, self.cfg.architecture.number_of_sets)).to(self.device)
                skeletons_block = []
                for ib, b in enumerate(block):
                    Xs, Ys, tokenized, xpr, equations = b
                    if isinstance(Xs, np.ndarray):  # Some blocks were stored as numpy arrays and others as tensors
                        Xs, Ys = torch.from_numpy(Xs), torch.from_numpy(Ys)
                    Xs = Xs.to(self.device)
                    Ys = Ys.to(self.device)
                    XY_block[ib, :, 0, :] = Xs[:, :self.cfg.architecture.number_of_sets]
                    XY_block[ib, :, 1, :] = Ys[:, :self.cfg.architecture.number_of_sets]
                    skeletons_block.append(torch.tensor(tokenized).long().to(self.device))
                    # print(xpr)

                self.model.set_train()  # Sets training mode
                running_loss = 0.0
                inds = np.arange(len(block))
                np.random.shuffle(inds)
                for step in range(T):  # Batch loop
                    # Zero the parameter gradients
                    self.optimizer.zero_grad()

                    # Generate indexes of the batch
                    batch_inds = inds[step * batch_size:(step + 1) * batch_size]
                    # Extract slices
                    XY_batch = XY_block[batch_inds, :, :, :]
                    skeletons_batch = [skeletons_block[i] for i in batch_inds]
                    # Check that there's no skeleton larger than the maximum length
                    valid_inds = [i for i in range(len(skeletons_batch)) if len(skeletons_batch[i]) < self.cfg.architecture.length_eq]
                    XY_batch = XY_batch[valid_inds, :, :, :]
                    skeletons_batch = [skeletons_batch[i] for i in valid_inds]

                    # Find the maximum skeleton length
                    max_length = max(len(sk) for sk in skeletons_batch)
                    # Pad the skeletons to match the maximum length
                    padded_tensors = [torch.cat((sk, torch.zeros(max_length - len(sk)).to(self.device))) for sk in skeletons_batch]
                    # Combine the padded skeletons into a single tensor
                    skeletons_batch = pad_sequence(padded_tensors, batch_first=True).type(torch.int)

                    # Forward pass
                    output, z_sets = self.model.forward(XY_batch, skeletons_batch)
                    # Loss calculation
                    L1 = torch.zeros(1).to(self.device)
                    for bi in range(output.shape[1]):
                        out = output[:, bi, :].contiguous().view(-1, output.shape[-1])
                        tokenized = skeletons_batch[bi, :][1:].contiguous().view(-1)
                        L1s = loss_sample(out, tokenized.long())
                        L1 += L1s
                    # Aggregate loss terms in the batch
                    loss = L1 / len(valid_inds)  # + self.lambda_ * L2) / (batch_size - skipped)
                    # Gradient computation
                    loss.backward()
                    # Optimization step
                    self.optimizer.step()

                    # Print statistics
                    batch_count += 1
                    global_batch_count += 1
                    running_loss += loss.item()
                    if batch_count % 5 == 0:
                        print('[%d, %5d] loss: %.5f' % (epoch + 1, batch_count, running_loss / 5))
                        self.writer.add_scalar('training loss', running_loss / 5, global_batch_count)
                        running_loss = 0.0

            #########################################################################
            # Validation step
            #########################################################################
            indexes = np.arange(len(val_files))
            batch_val_size = 50
            self.model.set_eval()
            L1v, L2v, iv = 0, 0, 0
            prev_loss = np.inf

            for b_ind in indexes:  # Block loop (each block contains 1000 inputs)
                # Read block
                block = open_pickle(val_files[b_ind])
                T = np.ceil(1.0 * len(block) / batch_val_size).astype(np.int32)

                # Format elements in the block as torch Tensors
                XY_block = torch.zeros((len(block), block[0][0].shape[0], 2, self.cfg.architecture.number_of_sets)).to(self.device)
                skeletons_block = []
                for ib, b in enumerate(block):
                    Xs, Ys, tokenized, xpr, equations = b
                    if isinstance(Xs, np.ndarray):  # Some blocks were stored as numpy arrays and others as tensors
                        Xs, Ys = torch.from_numpy(Xs), torch.from_numpy(Ys)
                    Xs = Xs.to(self.device)
                    Ys = Ys.to(self.device)
                    XY_block[ib, :, 0, :] = Xs[:, :self.cfg.architecture.number_of_sets]
                    XY_block[ib, :, 1, :] = Ys[:, :self.cfg.architecture.number_of_sets]
                    skeletons_block.append(torch.tensor(tokenized).long().to(self.device))

                inds = np.arange(len(block))
                for step in range(T):  # Batch loop
                    # Generate indexes of the batch
                    batch_inds = inds[step * batch_val_size:(step + 1) * batch_val_size]
                    # Extract slices
                    XY_batch = XY_block[batch_inds, :, :, :]
                    skeletons_batch = [skeletons_block[i] for i in batch_inds]
                    # Check that there's no skeleton larger than the maximum length
                    valid_inds = [i for i in range(len(skeletons_batch)) if len(skeletons_batch[i]) < self.cfg.architecture.length_eq]
                    XY_batch = XY_batch[valid_inds, :, :, :]
                    skeletons_batch = [skeletons_batch[i] for i in valid_inds]

                    # Find the maximum skeleton length
                    max_length = max(len(sk) for sk in skeletons_batch)
                    # Pad the skeletons to match the maximum length
                    padded_tensors = [torch.cat((sk, torch.zeros(max_length - len(sk)).to(self.device))) for sk in
                                      skeletons_batch]
                    # Combine the padded skeletons into a single tensor
                    skeletons_batch = pad_sequence(padded_tensors, batch_first=True).type(torch.int)
                    # Forward pass
                    # R = self.model.inference(XY_batch[0:1, :, :, :])
                    output = self.model.validation_step(XY_batch)
                    # Loss calculation
                    for bi in range(output.shape[1]):
                        out = output[:, bi, :].contiguous().view(-1, output.shape[-1])
                        tokenized = skeletons_batch[bi, :][1:].contiguous().view(-1)
                        padding_size = np.abs(output.size(0) - tokenized.size(0))
                        if output.size(0) > tokenized.size(0):
                            tokenized = nn.functional.pad(tokenized, (0, padding_size))
                        L1s = loss_sample(out, tokenized.long())
                        L1v += L1s
                        iv += 1
                        print("\tValidation " + str(iv), end='\r')

            # Aggregate loss terms in the batch
            loss = L1v / iv
            self.writer.add_scalar('validation loss', loss, global_batch_count)
            if loss < prev_loss:
                prev_loss = loss
                torch.save(self.model.state_dict(), 'src/EquationLearning/models/saved_models/Model1')
                with open('src/EquationLearning/models/saved_models/validation_performance.txt', 'w') as file:
                    file.write(str(loss))
            print('[%d] validation loss: %.5f. Best validation loss: %.5f' % (epoch + 1, loss, prev_loss))


if __name__ == '__main__':
    plt.figure()

    trainer = TransformerTrainer()
    trainer.fit()
