import sys
import os
from tensorflow.keras.utils import to_categorical

from warnings import filterwarnings

filterwarnings("ignore")

import utils
import utils_data

####################################### Settings #######################################
datasets = ['scenes', 'birds', 'flowers', 'cars', 'aircrafts']
n_classes = {'scenes': 67, 'flowers': 102, 'birds': 200, 'cars': 196, 'aircrafts': 90}

data_dir = sys.argv[1]
data_size = int(sys.argv[2])
valid_split = float(sys.argv[3])
test_split = float(sys.argv[4])
seed = int(sys.argv[5])
ds = sys.argv[6]
base_model_name = sys.argv[7]

result_filename = "result.txt"
data_shape = (data_size, data_size, 3)
batch_size = 16
data_augmentation = False

expert_weight_decays = [-1, 0.005, 0.0005]
expert_learning_rates = [0.01, 0.001, 0.0001]
expert_epsilons = [1e-07, 1e-08]
expert_epochs = 3
########################################################################################

for task_id, dataset in enumerate(datasets):
    if ds == dataset:
        utils.write_on_file(result_filename, 'w', "[%s] Loading data..." % dataset.upper())
        print("[%s] Loading data..." % dataset.upper())
        X_train, y_train, X_valid, y_valid, X_test, y_test = utils_data.load_data(data_dir=data_dir,
                                                                                  dataset=dataset,
                                                                                  data_size=data_size,
                                                                                  valid_split=valid_split,
                                                                                  test_split=test_split,
                                                                                  seed=seed)

        utils.write_on_file(result_filename, 'a', "[%s] Normalizing data..." % dataset.upper())
        print("[%s] Normalizing data..." % dataset.upper())
        X_train = utils.normalize_data(base_model_name=base_model_name, X=X_train)
        X_valid = utils.normalize_data(base_model_name=base_model_name, X=X_valid)
        X_test = utils.normalize_data(base_model_name=base_model_name, X=X_test)

        utils.write_on_file(result_filename, 'a', "[%s] Starting expert's cross-validation..." % dataset.upper())
        print("[%s] Starting expert's cross-validation..." % dataset.upper())
        expert = utils.expert_cross_validation(model_name=base_model_name,
                                               batch_size=batch_size,
                                               X_train=X_train, y_train=y_train,
                                               X_valid=X_valid, y_valid=y_valid,
                                               n_classes=n_classes[dataset],
                                               weight_decays=expert_weight_decays,
                                               learning_rates=expert_learning_rates,
                                               epsilons=expert_epsilons,
                                               epochs=expert_epochs,
                                               dataset=dataset,
                                               data_augmentation=data_augmentation)

        utils.write_on_file(result_filename, 'a', "[%s] Saving best expert..." % dataset.upper())
        print("[%s] Saving best expert..." % dataset.upper())
        expert.save(os.path.join('./', dataset + '_' + str(data_size) + '_' + base_model_name + '_expert.h5'))

        _, train_expert_acc = expert.evaluate(X_train, to_categorical(y_train), verbose=0)
        utils.write_on_file(result_filename, 'a', "[%s] TRAIN_EXPERT_ACC=%f" % (dataset.upper(), train_expert_acc))
        print("[%s] TRAIN_EXPERT_ACC=%f" % (dataset.upper(), train_expert_acc))

        _, valid_expert_acc = expert.evaluate(X_valid, to_categorical(y_valid), verbose=0)
        utils.write_on_file(result_filename, 'a', "[%s] VALID_EXPERT_ACC=%f" % (dataset.upper(), valid_expert_acc))
        print("[%s] VALID_EXPERT_ACC=%f" % (dataset.upper(), valid_expert_acc))

        _, test_expert_acc = expert.evaluate(X_test, to_categorical(y_test), verbose=0)
        utils.write_on_file(result_filename, 'a', "[%s] TEST_EXPERT_ACC=%f" % (dataset.upper(), test_expert_acc))
        print("[%s] TEST_EXPERT_ACC=%f\n" % (dataset.upper(), test_expert_acc))

utils.write_on_file(result_filename, 'a', "\n[INFO] Done")
print("\n[INFO] Done")
