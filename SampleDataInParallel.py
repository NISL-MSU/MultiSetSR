import omegaconf
from src.utils import *
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from src.EquationLearning.Transformers.GenerateTransformerData import Dataset, evaluate_and_wrap


def create_pickle_from_data(block, path, idx):
    with open(os.path.join(path, str(idx) + ".pkl"), 'wb') as file:
        pickle.dump(block, file)


def create_h5_from_data(block, path, idx):
    with h5py.File(os.path.join(path, str(idx) + ".h5"), "w") as hf:
        for ie, element in enumerate(block):
            # Create a group for each element using the index as the group name
            group = hf.create_group(str(ie))
            # Save "X" and "Y" as NumPy arrays
            group.create_dataset("X", data=element[0])
            group.create_dataset("Y", data=element[1])
            # Save 'tokenized' as a dataset of integers
            group.create_dataset("tokenized", data=element[2], dtype="i")
            # Save 'exprs' as a string
            group.create_dataset("exprs", data=np.string_(element[3]))
            # Convert sympy expressions to string and save 'sampled_exprs' as a list of strings
            sampled_exprs = [str(expr) for expr in element[4]]
            group.create_dataset("sampled_exprs", data=np.string_(sampled_exprs))


'''
Sample data for the multiple sets used to pre-train the Multi-set transformer.
It's located here to avoid problems with the ProcessPoolExecutor
'''


# Load configuration
cfg = omegaconf.OmegaConf.load("src/EquationLearning/Transformers/config.yaml")
data_train_path = cfg.train_path
data_val_path = cfg.val_path
training_dataset = Dataset(data_train_path, cfg.dataset_train, mode="train")
validation_dataset = Dataset(data_val_path, cfg.dataset_val, mode="val")
word2id = training_dataset.word2id


def process_data_point(pi):
    try:
        result = evaluate_and_wrap(validation_dataset[pi], cfg.dataset_train, word2id)
    except:
        result = None
    return result


if __name__ == '__main__':
    #####################################
    # Set the number of processes
    #####################################
    num_processes = 256  # 128 CPUs
    temp = []

    n_batch = 0
    for i in range(0, len(validation_dataset), 500):
        print('\n\t Starting batch. Sample ' + str(i))

        # Use process_data_point_with_timeout
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            try:
                result_gen = executor.map(process_data_point, range(i, i + 500), timeout=500)
                results = [x for x in result_gen]
            except TimeoutError:
                print('Got stuck, trying again' + str())
                try:
                    result_gen = executor.map(process_data_point, range(i, i + 500), timeout=500)
                    results = [x for x in result_gen]
                except TimeoutError:
                    continue

            print('\n\t Finished batch')
            print(len(results))
            print(sum(x is not None for x in results))

            # Process the results and create batches
            batch = []
            count = 0
            results = temp + results
            for step, sampled_data in enumerate(results):
                if sampled_data is not None:
                    count += 1
                    batch.append(sampled_data)
                    if count % 5000 == 0:
                        create_h5_from_data(batch, "src/EquationLearning/Data/sampled_data/" + cfg.dataset +
                                            "/validation", n_batch)
                        print("saving")
                        n_batch += 1
                        batch = []
            temp = batch.copy()  # Take the remaining samples
