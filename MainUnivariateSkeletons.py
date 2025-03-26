from EquationLearning.Trainer.TrainMultiSetTransformer import TransformerTrainer


if __name__ == '__main__':
    # Dataset names: E1 - E9, CS1-CS4
    trainer = TransformerTrainer()
    trainer.fit(pretrained=True)
