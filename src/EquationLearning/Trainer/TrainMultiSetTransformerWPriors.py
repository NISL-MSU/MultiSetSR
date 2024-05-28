import random
from src.EquationLearning.Trainer.TrainMultiSetTransformer import *


class TransformerTrainerwPrior(TransformerTrainer):
    """Pre-train transformer model using generated equations with prior information"""

    def get_unary_ops(self, tokenized):
        """Extract unary operators present in the expression"""
        prefix = de_tokenize(tokenized, self.id2word)
        un_ops, un_ops_tokenized = [], []
        for op, op_tokenized in zip(prefix, tokenized):
            if op in ['abs', 'acos', 'asin', 'atan', 'cos', 'cosh', 'div', 'exp', 'ln', 'pow', 'sin',
                      'sinh', 'sqrt', 'tan', 'tanh']:
                un_ops.append(op)
                un_ops_tokenized.append(op_tokenized)
        return un_ops, un_ops_tokenized

    def _init_model(self):
        """Initialize the Multi-Set Transformer with Priors model"""
        try:  # Read config yaml
            self.cfg = omegaconf.OmegaConf.load("src/EquationLearning/Transformers/config_withPrior.yaml")
        except FileNotFoundError:
            self.cfg = omegaconf.OmegaConf.load("../Transformers/config_withPrior.yaml")
        self._config_datasets()

        self.model_name = 'src/EquationLearning/models/saved_models/ModelWPriors480-batch_16-' + self.cfg.dataset

        return Model(cfg=self.cfg.architecture, cfg_inference=self.cfg.inference, word2id=self.word2id,
                     loss=loss_sample, priors=True)

    def _process_block(self, block):
        """ Format elements in the block as torch Tensors and remove inputs with NaN values"""
        XY_block = torch.zeros((len(block), self.cfg.architecture.block_size, 2, self.cfg.architecture.number_of_sets))
        un_ops_block = []
        skeletons_block = []
        xpr_block = []
        remove_indices = []
        for ib, b in enumerate(block):
            Xs, Ys, tokenized, xpr, equations = b
            Xs = Xs[:, :self.cfg.architecture.number_of_sets]
            Ys = Ys[:, :self.cfg.architecture.number_of_sets]
            Xs, Ys, _ = self.sample_domain(Xs, Ys, equations)

            # Extract all unary operators that appear in the expression and randomly sample some of them to act as priors
            _, all_un_ops = self.get_unary_ops(tokenized)
            if len(all_un_ops) == 0:
                un_ops = [0]
            elif len(all_un_ops) == 1:
                un_ops = all_un_ops
            else:
                un_ops = random.choices(all_un_ops, k=np.random.randint(1, len(all_un_ops)))  # Select a random subset

            # Shuffle data
            for d in range(self.cfg.architecture.number_of_sets):
                indices = np.arange(Xs.shape[0])
                np.random.shuffle(indices)
                Xs[:, d] = Xs[indices, d]
                Ys[:, d] = Ys[indices, d]
            # Normalize data
            means, std = np.mean(Ys, axis=0), np.std(Ys, axis=0)
            Ys = (Ys - means) / std

            if np.isnan(Ys).any() or np.min(std) < 0.01 or 'E' in xpr:
                remove_indices.append(ib)
            else:
                Xs, Ys = torch.from_numpy(Xs), torch.from_numpy(Ys)
                XY_block[ib, :, 0, :] = Xs
                XY_block[ib, :, 1, :] = Ys
                un_ops_block.append(torch.tensor(un_ops).long().cuda())
                skeletons_block.append(torch.tensor(tokenized).long().cuda())
                xpr_block.append(xpr)

        # Create a mask to exclude rows with specified indices
        mask = torch.ones(XY_block.shape[0], dtype=torch.bool, device=XY_block.device)
        mask[remove_indices] = 0
        # Use torch.index_select to select rows based on the mask
        XY_block = torch.index_select(XY_block, dim=0, index=mask.nonzero().squeeze()).cuda()

        return [XY_block, un_ops_block], skeletons_block, xpr_block

    def get_slices(self, input_block, skeletons_block, batch_inds):
        """Create a training batch by selecting certain indices from the data block"""
        XY_batch = input_block[0][batch_inds, :, :, :]
        un_ops_batch = [input_block[1][i] for i in batch_inds]
        skeletons_batch = [skeletons_block[i] for i in batch_inds]
        # Check that there's no skeleton larger than the maximum length
        valid_inds = [i for i in range(len(skeletons_batch)) if
                      len(skeletons_batch[i]) < self.cfg.architecture.length_eq]
        XY_batch = XY_batch[valid_inds, :, :, :]
        un_ops_batch = [un_ops_batch[i] for i in valid_inds]
        skeletons_batch = [skeletons_batch[i] for i in valid_inds]
        return [XY_batch, un_ops_batch], skeletons_batch, valid_inds


if __name__ == '__main__':
    trainer = TransformerTrainerwPrior()
    trainer.fit()
