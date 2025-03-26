import glob
import sympy
import torch
import warnings
import omegaconf
from torch import nn
from EquationLearning.utils import *
from torch import optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from EquationLearning.Transformers.model import Model
from EquationLearning.Transformers.GenerateTransformerData import Dataset, de_tokenize


def open_pickle(path):
    with open(path, 'rb') as file:
        block = pickle.load(file)
    return block


def open_h5(path):
    block = []
    with h5py.File(path, "r") as hf:
        # Iterate through the groups in the HDF5 file (group names are integers)
        for group_name in hf:
            group = hf[group_name]
            # Read data from the group
            X = group["X"][:]
            Y = group["Y"][:]
            # Load 'tokenized' as a list of integers
            tokenized = list(group["tokenized"])
            # Load 'exprs' as a string
            exprs = group["exprs"][()].tobytes().decode("utf-8")
            # Load 'sampled_exprs' as a list of sympy expressions
            sampled_exprs = [expr_str for expr_str in group["sampled_exprs"][:].astype(str)]
            block.append([X, Y, tokenized, exprs, sampled_exprs])
    return block


def seq2equation(tokenized, id2word, printFlag=False):
    prefix = de_tokenize(tokenized, id2word)
    if printFlag:
        print("Prefix notation: " + str(prefix))
    env, param, config_dict = create_env(os.path.join(get_project_root(), "EquationLearning//dataset_configuration.json"))
    infix = env.prefix_to_infix(prefix, coefficients=env.coefficients, variables=env.variables)
    return infix


def loss_sample(output, trg, operators_tokens, prior_ops=None, penalty_factor=0.5):
    """Loss function for a single sample"""
    ce = nn.CrossEntropyLoss(ignore_index=0)
    ce.cuda()
    if prior_ops is None:
        return ce(output, trg)
    else:
        _, predicted_tokens = torch.max(output, dim=1)

        predicted_set = set(predicted_tokens.tolist())
        prior_ops_set = set(prior_ops)
        total_ops_set = set(operators_tokens)

        # Calculate penalization
        penalized_tokens = predicted_set.intersection(total_ops_set) - prior_ops_set
        penalization = len(penalized_tokens) * penalty_factor

        return ce(output, trg) + penalization


class TransformerTrainer:
    """Pre-train transformer model using generated equations"""

    def __init__(self):
        """
        Initialize TransformerTrainer class
        """
        # Init model
        self.model = self._init_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.cuda()
        if torch.cuda.device_count() > 1:
            print("Using ", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)

        # Training parameters
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=self.cfg.architecture.lr)
        self.writer = SummaryWriter('runs')
        self.lambda_ = self.cfg.dataset_train.lambda_

    def _config_datasets(self):
        # Configuration datasets
        self.sampledData_train_path = os.path.join(get_project_root(), 'EquationLearning/Data/sampled_data/' + self.cfg.dataset + '/training')
        self.sampledData_val_path = os.path.join(get_project_root(), 'EquationLearning/Data/sampled_data/' + self.cfg.dataset + '/validation')
        self.data_train_path = self.cfg.train_path
        self.training_dataset = Dataset(self.data_train_path, self.cfg.dataset_train, mode="train")
        self.word2id = self.training_dataset.word2id
        self.id2word = self.training_dataset.id2word
        # Extract the tokens corresponding to mathematical operators
        self.operators_tokens = [self.word2id[n] for n in list(self.word2id.keys()) if not (n.isnumeric() or n == 'x_1' or len(n) == 1)]

    def _init_model(self):
        try:  # Read config yaml
            self.cfg = omegaconf.OmegaConf.load("EquationLearning/Transformers/config.yaml")
        except FileNotFoundError:
            self.cfg = omegaconf.OmegaConf.load("src/EquationLearning/Transformers/config.yaml")
        self._config_datasets()

        # Name structure: "Model{dim_hidden}-batch_{batch_size}-{dataset_name}"
        self.model_name = 'src/EquationLearning/saved_models/saved_MSTs/Model' + str(self.cfg.architecture.dim_hidden) + \
                          '-batch_' + str(self.cfg.batch_size) + '-' + self.cfg.dataset

        return Model(cfg=self.cfg.architecture, cfg_inference=self.cfg.inference, word2id=self.word2id,
                     loss=loss_sample)

    def load_model(self, pretrained):
        # Load pre-trained weights
        if os.path.exists(self.model_name) and pretrained:
            if torch.cuda.device_count() > 1:
                self.model.module.load_state_dict(torch.load(self.model_name))
            else:
                self.model.load_state_dict(torch.load(self.model_name))
        else:
            warnings.warn('There was no model saved. Start training from scratch...')

    def sample_domain(self, Xs, Ys, equations):
        """Use a random domain (e.g., between -10 and 10, or -5 and 5, etc)"""
        dva = np.random.randint(3, 10)
        X, Y = np.zeros((self.cfg.architecture.block_size, self.cfg.architecture.number_of_sets)), np.zeros(
            (self.cfg.architecture.block_size, self.cfg.architecture.number_of_sets))
        ns = 0
        while ns < self.cfg.architecture.number_of_sets:
            minX, maxX = -dva, dva
            # Select rows where the value of the first column is between minX and maxX
            selected_rows_indices = np.where((Xs[:, ns] >= minX) & (Xs[:, ns] <= maxX))[0]
            remaining = self.cfg.architecture.block_size - len(selected_rows_indices)
            # Randomly select 'self.cfg.architecture.block_size' rows from the selected rows
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
            if np.min(np.std(Y, axis=0)) < 0.0001 and dva < 10:
                ns, dva = 0, dva + 1  # If the selected domain is too flat, try with a larger one
                continue
            ns += 1
        # With a chance of 0.3, fix all sets to the same function
        if np.random.random(1) < 0.3:
            ns = np.random.randint(0, self.cfg.architecture.number_of_sets)
            X[:, 0:] = X[:, ns][:, np.newaxis]
            Y[:, 0:] = Y[:, ns][:, np.newaxis]
            equations = [equations[ns]] * self.cfg.architecture.number_of_sets
        return X, Y, equations

    def _process_block(self, block):
        """ Format elements in the block as torch Tensors and remove inputs with NaN values"""
        XY_block = torch.zeros((len(block), self.cfg.architecture.block_size, 2, self.cfg.architecture.number_of_sets))
        skeletons_block = []
        xpr_block = []
        remove_indices = []
        for ib, b in enumerate(block):
            Xs, Ys, tokenized, xpr, equations = b
            Xs = Xs[:, :self.cfg.architecture.number_of_sets]
            Ys = Ys[:, :self.cfg.architecture.number_of_sets]
            Xs, Ys, _ = self.sample_domain(Xs, Ys, equations)

            # Shuffle data
            for d in range(self.cfg.architecture.number_of_sets):
                indices = np.arange(Xs.shape[0])
                np.random.shuffle(indices)
                Xs[:, d] = Xs[indices, d]
                Ys[:, d] = Ys[indices, d]
            # Normalize data
            means, std = np.mean(Ys, axis=0), np.std(Ys, axis=0)
            Ys = (Ys - means) / std

            if np.isnan(Ys).any() or np.min(std) < 0.0001 or 'E' in xpr:
                remove_indices.append(ib)
            else:
                Xs, Ys = torch.from_numpy(Xs), torch.from_numpy(Ys)
                XY_block[ib, :, 0, :] = Xs
                XY_block[ib, :, 1, :] = Ys
                skeletons_block.append(torch.tensor(tokenized).long().cuda())
                xpr_block.append(xpr)

        # Create a mask to exclude rows with specified indices
        mask = torch.ones(XY_block.shape[0], dtype=torch.bool, device=XY_block.device)
        mask[remove_indices] = 0
        # Use torch.index_select to select rows based on the mask
        XY_block = torch.index_select(XY_block, dim=0, index=mask.nonzero().squeeze()).cuda()

        return XY_block, skeletons_block, xpr_block

    def get_slices(self, input_block, skeletons_block, batch_inds):
        """Create a training batch by selecting certain indices from the data block"""
        XY_batch = input_block[batch_inds, :, :, :]
        skeletons_batch = [skeletons_block[i] for i in batch_inds]
        # Check that there's no skeleton larger than the maximum length
        valid_inds = [i for i in range(len(skeletons_batch)) if len(skeletons_batch[i]) < self.cfg.architecture.length_eq]

        if torch.cuda.device_count() > 1:  # Ensure len(valid_inds) is a multiple of 4
            valid_len = len(valid_inds)
            if valid_len % 4 != 0:
                valid_len = (valid_len // 4) * 4
            valid_inds = valid_inds[:valid_len]
        XY_batch = XY_batch[valid_inds, :, :, :]
        skeletons_batch = [skeletons_batch[i] for i in valid_inds]
        return XY_batch, skeletons_batch, valid_inds

    def fit(self, pretrained: bool = False):
        """Implement main training loop
        :param pretrained: If True, the weights of the model are initialize loading the weights of a previously trained model"""
        epochs = self.cfg.epochs
        batch_size = self.cfg.batch_size
        # Get names of training and val blocks
        train_files = glob.glob(os.path.join(self.sampledData_train_path, '*.h5'))
        val_files = glob.glob(os.path.join(self.sampledData_val_path, '*.h5'))
        # Prepare list of indexes for shuffling
        indexes = np.arange(len(train_files))

        # Load pre-trained weights
        self.load_model(pretrained=pretrained)

        print("""""""""""""""""""""""""""""")
        print("Start training")
        print("""""""""""""""""""""""""""""")
        global_batch_count = 0
        prev_loss = np.inf
        for epoch in range(epochs):  # Epoch loop
            np.random.shuffle(indexes)

            batch_count = 0
            for b_ind in indexes:  # Block loop
                # Read block
                block = open_h5(train_files[b_ind])
                input_block, skeletons_block, xpr_block = self._process_block(block)

                if torch.cuda.device_count() > 1:
                    self.model.module.set_train()   # Sets training mode
                else:
                    self.model.set_train()          # Sets training mode
                running_loss = 0.0
                inds = np.arange(len(skeletons_block))
                np.random.shuffle(inds)
                T = np.ceil(1.0 * len(skeletons_block) / batch_size).astype(np.int32)
                for step in range(T):  # Batch loop
                    # Generate indexes of the batch
                    batch_inds = inds[step * batch_size:(step + 1) * batch_size]
                    print("Block " + str(train_files[b_ind]) + " Sample " + str(batch_inds[0]) + " Expr: " + str(xpr_block[batch_inds[0]]))
                    input_batch, skeletons_batch, valid_inds = self.get_slices(input_block, skeletons_block, batch_inds)

                    # Find the maximum skeleton length
                    max_length = max(len(sk) for sk in skeletons_batch)
                    # Pad the skeletons to match the maximum length
                    padded_tensors = [torch.cat((sk, torch.zeros(max_length - len(sk)).cuda())) for sk in
                                      skeletons_batch]
                    # Combine the padded skeletons into a single tensor
                    skeletons_batch = pad_sequence(padded_tensors, batch_first=True).type(torch.int).cuda()

                    # Zero the parameter gradients
                    self.optimizer.zero_grad()

                    # Forward pass
                    if torch.cuda.device_count() > 1:
                        output, z_sets, L1 = self.model.forward(input_batch, skeletons_batch.cuda())
                        # Aggregate loss terms in the batch
                        L1 = L1.sum()
                    else:
                        output, z_sets = self.model.forward(input_batch, skeletons_batch.cuda())
                        # Loss calculation
                        L1 = torch.zeros(1).cuda()
                        for bi in range(output.shape[1]):
                            out = output[:, bi, :].contiguous().view(-1, output.shape[-1])
                            tokenized = skeletons_batch[bi, :][1:].contiguous().view(-1)
                            L1s = loss_sample(out, tokenized.long(), operators_tokens=None)
                            L1 += L1s

                    loss = L1 / len(valid_inds)
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

            if epoch == 0:  # Save model at the end of the first epoch in case there's an error during validation
                if torch.cuda.device_count() > 1:
                    torch.save(self.model.module.state_dict(), self.model_name)
                else:
                    torch.save(self.model.state_dict(), self.model_name)
            #########################################################################
            # Validation step
            #########################################################################
            indexes2 = np.arange(len(val_files))
            batch_val_size = 20
            if torch.cuda.device_count() > 1:
                self.model.module.set_eval()
            else:
                self.model.set_eval()
            L1v, L2v, iv = 0, 0, 0

            cc = 0
            np.random.shuffle(indexes2)
            for b_ind in indexes2:  # Block loop (each block contains 1000 inputs)
                # Read block
                block = open_h5(val_files[b_ind])
                input_block, skeletons_block, xpr_block = self._process_block(block)

                inds = np.arange(len(skeletons_block))
                T = np.ceil(1.0 * len(skeletons_block) / batch_val_size).astype(np.int32)
                for step in range(T):  # Batch loop
                    # Generate indexes of the batch
                    batch_inds = inds[step * batch_val_size:(step + 1) * batch_val_size]
                    input_batch, skeletons_batch, valid_inds = self.get_slices(input_block, skeletons_block, batch_inds)

                    # Find the maximum skeleton length
                    max_length = max(len(sk) for sk in skeletons_batch)
                    # Pad the skeletons to match the maximum length
                    padded_tensors = [torch.cat((sk, torch.zeros(max_length - len(sk)).cuda())) for sk in
                                      skeletons_batch]
                    # Combine the padded skeletons into a single tensor
                    skeletons_batch = pad_sequence(padded_tensors, batch_first=True).type(torch.int)
                    # Forward pass
                    if torch.cuda.device_count() > 1:
                        output = self.model.module.validation_step(input_batch, skeletons_batch)
                    else:
                        output = self.model.validation_step(input_batch, skeletons_batch)
                    # Loss calculation
                    for bi in range(output.shape[1]):
                        out = output[:, bi, :].contiguous().view(-1, output.shape[-1])
                        tokenized = skeletons_batch[bi, :][1:].contiguous().view(-1)
                        L1s = loss_sample(out, tokenized.long(), operators_tokens=None)
                        L1v += L1s
                        iv += 1
                        res = output.cpu().numpy()[:, 0, :]
                        max_indices = np.argmax(res, axis=1)
                        try:
                            infix = sympy.sympify(seq2equation(max_indices, self.id2word))
                            infixT = sympy.sympify(
                                seq2equation(list(skeletons_block[step].cpu().numpy())[1:], self.id2word))
                            print("Step: " + str(step) + "Target: " + str(infixT) + " . Pred: " + str(infix))
                        except:
                            continue
                cc += 1
                with open('EquationLearning/saved_models/saved_MSTs/validation_performance.txt', 'w') as file:
                    file.write(str(L1v / (5000 * cc)))

            # Aggregate loss terms in the batch
            loss = L1v / iv
            self.writer.add_scalar('validation loss', loss, global_batch_count)
            if loss < prev_loss:
                prev_loss = loss
                if torch.cuda.device_count() > 1:
                    torch.save(self.model.module.state_dict(), self.model_name)
                else:
                    torch.save(self.model.state_dict(), self.model_name)
                with open('EquationLearning/saved_models/saved_MSTs/validation_performance.txt', 'w') as file:
                    file.write(str(loss))
            print('[%d] validation loss: %.5f. Best validation loss: %.5f' % (epoch + 1, loss, prev_loss))


if __name__ == '__main__':
    trainer = TransformerTrainer()
    trainer.fit()
