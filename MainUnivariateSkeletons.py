from EquationLearning.Trainer.TrainMultiSetTransformerWPriors import TransformerTrainerwPrior


if __name__ == '__main__':
    # Dataset names: E1 - E9, CS1-CS4
    trainer = TransformerTrainerwPrior()
    trainer.fit(pretrained=True)
