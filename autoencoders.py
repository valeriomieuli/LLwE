import os
import sys
import numpy as np
from warnings import filterwarnings

filterwarnings("ignore")

import utils
import utils_data
import utils_models

####################################### Settings #######################################
datasets = ['scenes', 'birds', 'flowers', 'cars', 'aircrafts']
n_classes = {'scenes': 67, 'flowers': 102, 'birds': 200, 'cars': 196, 'aircrafts': 90}

data_dir = sys.argv[1]
data_size = int(sys.argv[2])
valid_split = float(sys.argv[3])
test_split = float(sys.argv[4])
seed = int(sys.argv[5])
ds = sys.argv[6]

result_filename = "result.txt"
data_shape = (data_size, data_size, 3)
batch_size = 16

features_extractor_name = 'VGG16'
autoencoder_hidden_layer_sizes = [50, 100, 200]
autoencoder_weight_decays = [-1, 0.005, 0.0005]
autoencoder_learning_rates = [0.1, 0.01, 0.001]
autoencoder_epsilons = [1e-07, 1e-08]
autoencoder_objective_loss = 'binary_crossentropy'
autoencoder_epochs = 20
########################################################################################

features_mean = np.load(os.path.join(os.getcwd(), str(data_size) + '_mean.npy'))
features_std = np.load(os.path.join(os.getcwd(), str(data_size) + '_std.npy'))
features_extractor = utils_models.build_features_extractor(model_name=features_extractor_name, input_shape=data_shape)

for task_id, dataset in enumerate(datasets):
    if ds == dataset:
        utils.write_on_file(result_filename, 'w', "[%s] Loading data..." % dataset.upper())
        print("[%s] Loading data..." % dataset.upper())
        X_train, _, X_valid, _, _, _ = utils_data.load_data(data_dir=data_dir,
                                                            dataset=dataset,
                                                            data_size=data_size,
                                                            valid_split=valid_split,
                                                            test_split=test_split,
                                                            seed=seed)

        utils.write_on_file(result_filename, 'a', "[%s] Starting autoencoder's cross-validation..." % dataset.upper())
        print("[%s] Starting autoencoder's cross-validation..." % dataset.upper())
        autoencoder = utils.autoencoder_cross_validation(features_extractor=features_extractor,
                                                         batch_size=batch_size, X_train=X_train, X_valid=X_valid,
                                                         features_mean=features_mean, features_std=features_std,
                                                         hidden_layer_sizes=autoencoder_hidden_layer_sizes,
                                                         weight_decays=autoencoder_weight_decays,
                                                         learning_rates=autoencoder_learning_rates,
                                                         epsilons=autoencoder_epsilons, epochs=autoencoder_epochs,
                                                         objective_loss=autoencoder_objective_loss, dataset=ds)

        utils.write_on_file(result_filename, 'a', "[%s] Saving best autoencoder..." % dataset.upper())
        print("[%s] Saving best autoencoder..." % dataset.upper())
        autoencoder.save(os.path.join('./', dataset + '_' + str(data_size) + '_autoencoder.h5'))

utils.write_on_file(result_filename, 'a', "\n[INFO] Done")
print("\n[INFO] Done")
