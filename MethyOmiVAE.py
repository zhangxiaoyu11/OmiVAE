import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from earlystoping import Earlystopping
from sklearn import metrics


def MethyOmiVAE(input_path, methy_chr_df_list, random_seed=42, no_cuda=False, model_parallelism=True,
                separate_testing=True, batch_size=32, latent_dim=128, learning_rate=1e-3, p1_epoch_num=50,
                p2_epoch_num=100, output_loss_record=True, classifier=True, early_stopping=True):

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    device = torch.device('cuda:0' if not no_cuda and torch.cuda.is_available() else 'cpu')
    parallel = torch.cuda.device_count() > 1 and model_parallelism

    sample_id = np.loadtxt(input_path + 'both_samples.tsv', delimiter='\t', dtype='str')

    # Loading label
    label = pd.read_csv(input_path + 'both_samples_tumour_type_digit.tsv', sep='\t', header=0, index_col=0)
    class_num = len(label.tumour_type.unique())
    label_array = label['tumour_type'].values

    if separate_testing:
        # Get testing set index and training set index
        # Separate according to different tumour types
        testset_ratio = 0.2
        valset_ratio = 0.5

        train_index, test_index, train_label, test_label = train_test_split(sample_id, label_array,
                                                                            test_size=testset_ratio,
                                                                            random_state=random_seed,
                                                                            stratify=label_array)
        val_index, test_index, val_label, test_label = train_test_split(test_index, test_label, test_size=valset_ratio,
                                                                        random_state=random_seed, stratify=test_label)

        methy_chr_df_test_list = []
        methy_chr_df_val_list = []
        methy_chr_df_train_list = []
        for chrom_index in range(0, 23):
            methy_chr_df_test = methy_chr_df_list[chrom_index][test_index]
            methy_chr_df_test_list.append(methy_chr_df_test)
            methy_chr_df_val = methy_chr_df_list[chrom_index][val_index]
            methy_chr_df_val_list.append(methy_chr_df_val)
            methy_chr_df_train = methy_chr_df_list[chrom_index][train_index]
            methy_chr_df_train_list.append(methy_chr_df_train)

    # Get dataset information
    sample_num = len(sample_id)
    methy_feature_num_list = []
    for chrom_index in range(0, 23):
        feature_num = methy_chr_df_list[chrom_index].shape[0]
        methy_feature_num_list.append(feature_num)
    methy_feature_num_array = np.array(methy_feature_num_list)
    methy_feature_num = methy_feature_num_array.sum()
    print('\nNumber of samples: {}'.format(sample_num))
    print('Number of methylation features: {}'.format(methy_feature_num))
    if classifier:
        print('Number of classes: {}'.format(class_num))

    class MethyOmiDataset(Dataset):
        def __init__(self, methy_df_list, labels):
            self.methy_df_list = methy_df_list
            self.labels = labels

        def __len__(self):
            return self.methy_df_list[0].shape[1]

        def __getitem__(self, index):
            omics_data = []
            # Methylation tensor index 0-22
            for methy_chrom_index in range(0, 23):
                methy_chr_line = self.methy_df_list[methy_chrom_index].iloc[:, index].values
                methy_chr_line_tensor = torch.Tensor(methy_chr_line)
                omics_data.append(methy_chr_line_tensor)
            label = self.labels[index]
            return [omics_data, label]

    # DataSets and DataLoaders
    if separate_testing:
        train_dataset = MethyOmiDataset(methy_df_list=methy_chr_df_train_list, labels=train_label)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
        val_dataset = MethyOmiDataset(methy_df_list=methy_chr_df_val_list, labels=val_label)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
        test_dataset = MethyOmiDataset(methy_df_list=methy_chr_df_test_list, labels=test_label)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    else:
        train_dataset = MethyOmiDataset(methy_df_list=methy_chr_df_list)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    full_dataset = MethyOmiDataset(methy_df_list=methy_chr_df_list, labels=label_array)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, num_workers=6)

    # Setting dimensions
    latent_space_dim = latent_dim
    input_dim_methy_array = methy_feature_num_array
    level_2_dim_methy = 256
    level_3_dim_methy = 1024
    level_4_dim = 512
    classifier_1_dim = 128
    classifier_2_dim = 64
    classifier_out_dim = class_num

    class VAE(nn.Module):
        def __init__(self):
            super(VAE, self).__init__()
            # ENCODER fc layers
            # level 1
            # Methy input for each chromosome
            self.e_fc1_methy_1 = self.fc_layer(input_dim_methy_array[0], level_2_dim_methy)
            self.e_fc1_methy_2 = self.fc_layer(input_dim_methy_array[1], level_2_dim_methy)
            self.e_fc1_methy_3 = self.fc_layer(input_dim_methy_array[2], level_2_dim_methy)
            self.e_fc1_methy_4 = self.fc_layer(input_dim_methy_array[3], level_2_dim_methy)
            self.e_fc1_methy_5 = self.fc_layer(input_dim_methy_array[4], level_2_dim_methy)
            self.e_fc1_methy_6 = self.fc_layer(input_dim_methy_array[5], level_2_dim_methy)
            self.e_fc1_methy_7 = self.fc_layer(input_dim_methy_array[6], level_2_dim_methy)
            self.e_fc1_methy_8 = self.fc_layer(input_dim_methy_array[7], level_2_dim_methy)
            self.e_fc1_methy_9 = self.fc_layer(input_dim_methy_array[8], level_2_dim_methy)
            self.e_fc1_methy_10 = self.fc_layer(input_dim_methy_array[9], level_2_dim_methy)
            self.e_fc1_methy_11 = self.fc_layer(input_dim_methy_array[10], level_2_dim_methy)
            self.e_fc1_methy_12 = self.fc_layer(input_dim_methy_array[11], level_2_dim_methy)
            self.e_fc1_methy_13 = self.fc_layer(input_dim_methy_array[12], level_2_dim_methy)
            self.e_fc1_methy_14 = self.fc_layer(input_dim_methy_array[13], level_2_dim_methy)
            self.e_fc1_methy_15 = self.fc_layer(input_dim_methy_array[14], level_2_dim_methy)
            self.e_fc1_methy_16 = self.fc_layer(input_dim_methy_array[15], level_2_dim_methy)
            self.e_fc1_methy_17 = self.fc_layer(input_dim_methy_array[16], level_2_dim_methy)
            self.e_fc1_methy_18 = self.fc_layer(input_dim_methy_array[17], level_2_dim_methy)
            self.e_fc1_methy_19 = self.fc_layer(input_dim_methy_array[18], level_2_dim_methy)
            self.e_fc1_methy_20 = self.fc_layer(input_dim_methy_array[19], level_2_dim_methy)
            self.e_fc1_methy_21 = self.fc_layer(input_dim_methy_array[20], level_2_dim_methy)
            self.e_fc1_methy_22 = self.fc_layer(input_dim_methy_array[21], level_2_dim_methy)
            self.e_fc1_methy_X = self.fc_layer(input_dim_methy_array[22], level_2_dim_methy)

            # Level 2
            self.e_fc2_methy = self.fc_layer(level_2_dim_methy*23, level_3_dim_methy)
            # self.e_fc2_methy = self.fc_layer(level_2_dim_methy * 23, level_3_dim_methy, dropout=True)

            # Level 3
            self.e_fc3 = self.fc_layer(level_3_dim_methy, level_4_dim)
            # self.e_fc3 = self.fc_layer(level_3_dim_methy, level_4_dim, dropout=True)

            # Level 4
            self.e_fc4_mean = self.fc_layer(level_4_dim, latent_space_dim, activation=0)
            self.e_fc4_log_var = self.fc_layer(level_4_dim, latent_space_dim, activation=0)

            # model parallelism
            if parallel:
                self.e_fc1_methy_1.to('cuda:0')
                self.e_fc1_methy_2.to('cuda:0')
                self.e_fc1_methy_3.to('cuda:0')
                self.e_fc1_methy_4.to('cuda:0')
                self.e_fc1_methy_5.to('cuda:0')
                self.e_fc1_methy_6.to('cuda:0')
                self.e_fc1_methy_7.to('cuda:0')
                self.e_fc1_methy_8.to('cuda:0')
                self.e_fc1_methy_9.to('cuda:0')
                self.e_fc1_methy_10.to('cuda:0')
                self.e_fc1_methy_11.to('cuda:0')
                self.e_fc1_methy_12.to('cuda:0')
                self.e_fc1_methy_13.to('cuda:0')
                self.e_fc1_methy_14.to('cuda:0')
                self.e_fc1_methy_15.to('cuda:0')
                self.e_fc1_methy_16.to('cuda:0')
                self.e_fc1_methy_17.to('cuda:0')
                self.e_fc1_methy_18.to('cuda:0')
                self.e_fc1_methy_19.to('cuda:0')
                self.e_fc1_methy_20.to('cuda:0')
                self.e_fc1_methy_21.to('cuda:0')
                self.e_fc1_methy_22.to('cuda:0')
                self.e_fc1_methy_X.to('cuda:0')
                self.e_fc2_methy.to('cuda:0')
                self.e_fc3.to('cuda:0')
                self.e_fc4_mean.to('cuda:0')
                self.e_fc4_log_var.to('cuda:0')

            # DECODER fc layers
            # Level 4
            self.d_fc4 = self.fc_layer(latent_space_dim, level_4_dim)

            # Level 3
            self.d_fc3 = self.fc_layer(level_4_dim, level_3_dim_methy)
            # self.d_fc3 = self.fc_layer(level_4_dim, level_3_dim_methy, dropout=True)

            # Level 2
            self.d_fc2_methy = self.fc_layer(level_3_dim_methy, level_2_dim_methy*23)
            # self.d_fc2_methy = self.fc_layer(level_3_dim_methy, level_2_dim_methy*23, dropout=True)

            # level 1
            # Methy output for each chromosome
            self.d_fc1_methy_1 = self.fc_layer(level_2_dim_methy, input_dim_methy_array[0], activation=2)
            self.d_fc1_methy_2 = self.fc_layer(level_2_dim_methy, input_dim_methy_array[1], activation=2)
            self.d_fc1_methy_3 = self.fc_layer(level_2_dim_methy, input_dim_methy_array[2], activation=2)
            self.d_fc1_methy_4 = self.fc_layer(level_2_dim_methy, input_dim_methy_array[3], activation=2)
            self.d_fc1_methy_5 = self.fc_layer(level_2_dim_methy, input_dim_methy_array[4], activation=2)
            self.d_fc1_methy_6 = self.fc_layer(level_2_dim_methy, input_dim_methy_array[5], activation=2)
            self.d_fc1_methy_7 = self.fc_layer(level_2_dim_methy, input_dim_methy_array[6], activation=2)
            self.d_fc1_methy_8 = self.fc_layer(level_2_dim_methy, input_dim_methy_array[7], activation=2)
            self.d_fc1_methy_9 = self.fc_layer(level_2_dim_methy, input_dim_methy_array[8], activation=2)
            self.d_fc1_methy_10 = self.fc_layer(level_2_dim_methy, input_dim_methy_array[9], activation=2)
            self.d_fc1_methy_11 = self.fc_layer(level_2_dim_methy, input_dim_methy_array[10], activation=2)
            self.d_fc1_methy_12 = self.fc_layer(level_2_dim_methy, input_dim_methy_array[11], activation=2)
            self.d_fc1_methy_13 = self.fc_layer(level_2_dim_methy, input_dim_methy_array[12], activation=2)
            self.d_fc1_methy_14 = self.fc_layer(level_2_dim_methy, input_dim_methy_array[13], activation=2)
            self.d_fc1_methy_15 = self.fc_layer(level_2_dim_methy, input_dim_methy_array[14], activation=2)
            self.d_fc1_methy_16 = self.fc_layer(level_2_dim_methy, input_dim_methy_array[15], activation=2)
            self.d_fc1_methy_17 = self.fc_layer(level_2_dim_methy, input_dim_methy_array[16], activation=2)
            self.d_fc1_methy_18 = self.fc_layer(level_2_dim_methy, input_dim_methy_array[17], activation=2)
            self.d_fc1_methy_19 = self.fc_layer(level_2_dim_methy, input_dim_methy_array[18], activation=2)
            self.d_fc1_methy_20 = self.fc_layer(level_2_dim_methy, input_dim_methy_array[19], activation=2)
            self.d_fc1_methy_21 = self.fc_layer(level_2_dim_methy, input_dim_methy_array[20], activation=2)
            self.d_fc1_methy_22 = self.fc_layer(level_2_dim_methy, input_dim_methy_array[21], activation=2)
            self.d_fc1_methy_X = self.fc_layer(level_2_dim_methy, input_dim_methy_array[22], activation=2)

            # model parallelism
            if parallel:
                self.d_fc4.to('cuda:1')
                self.d_fc3.to('cuda:1')
                self.d_fc2_methy.to('cuda:1')
                self.d_fc1_methy_1.to('cuda:1')
                self.d_fc1_methy_2.to('cuda:1')
                self.d_fc1_methy_3.to('cuda:1')
                self.d_fc1_methy_4.to('cuda:1')
                self.d_fc1_methy_5.to('cuda:1')
                self.d_fc1_methy_6.to('cuda:1')
                self.d_fc1_methy_7.to('cuda:1')
                self.d_fc1_methy_8.to('cuda:1')
                self.d_fc1_methy_9.to('cuda:1')
                self.d_fc1_methy_10.to('cuda:1')
                self.d_fc1_methy_11.to('cuda:1')
                self.d_fc1_methy_12.to('cuda:1')
                self.d_fc1_methy_13.to('cuda:1')
                self.d_fc1_methy_14.to('cuda:1')
                self.d_fc1_methy_15.to('cuda:1')
                self.d_fc1_methy_16.to('cuda:1')
                self.d_fc1_methy_17.to('cuda:1')
                self.d_fc1_methy_18.to('cuda:1')
                self.d_fc1_methy_19.to('cuda:1')
                self.d_fc1_methy_20.to('cuda:1')
                self.d_fc1_methy_21.to('cuda:1')
                self.d_fc1_methy_22.to('cuda:1')
                self.d_fc1_methy_X.to('cuda:1')

            # CLASSIFIER fc layers
            self.c_fc1 = self.fc_layer(latent_space_dim, classifier_1_dim)
            self.c_fc2 = self.fc_layer(classifier_1_dim, classifier_2_dim)
            # self.c_fc2 = self.fc_layer(classifier_1_dim, classifier_2_dim, dropout=True)
            self.c_fc3 = self.fc_layer(classifier_2_dim, classifier_out_dim, activation=0)

            # model parallelism
            if parallel:
                self.c_fc1.to('cuda:1')
                self.c_fc2.to('cuda:1')
                self.c_fc3.to('cuda:1')

        # Activation - 0: no activation, 1: ReLU, 2: Sigmoid
        def fc_layer(self, in_dim, out_dim, activation=1, dropout=False, dropout_p=0.5):
            if activation == 0:
                layer = nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim))
            elif activation == 2:
                layer = nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.Sigmoid())
            else:
                if dropout:
                    layer = nn.Sequential(
                        nn.Linear(in_dim, out_dim),
                        nn.BatchNorm1d(out_dim),
                        nn.ReLU(),
                        nn.Dropout(p=dropout_p))
                else:
                    layer = nn.Sequential(
                        nn.Linear(in_dim, out_dim),
                        nn.BatchNorm1d(out_dim),
                        nn.ReLU())
            return layer

        def encode(self, x):
            methy_1_level2_layer = self.e_fc1_methy_1(x[0])
            methy_2_level2_layer = self.e_fc1_methy_2(x[1])
            methy_3_level2_layer = self.e_fc1_methy_3(x[2])
            methy_4_level2_layer = self.e_fc1_methy_4(x[3])
            methy_5_level2_layer = self.e_fc1_methy_5(x[4])
            methy_6_level2_layer = self.e_fc1_methy_6(x[5])
            methy_7_level2_layer = self.e_fc1_methy_7(x[6])
            methy_8_level2_layer = self.e_fc1_methy_8(x[7])
            methy_9_level2_layer = self.e_fc1_methy_9(x[8])
            methy_10_level2_layer = self.e_fc1_methy_10(x[9])
            methy_11_level2_layer = self.e_fc1_methy_11(x[10])
            methy_12_level2_layer = self.e_fc1_methy_12(x[11])
            methy_13_level2_layer = self.e_fc1_methy_13(x[12])
            methy_14_level2_layer = self.e_fc1_methy_14(x[13])
            methy_15_level2_layer = self.e_fc1_methy_15(x[14])
            methy_16_level2_layer = self.e_fc1_methy_16(x[15])
            methy_17_level2_layer = self.e_fc1_methy_17(x[16])
            methy_18_level2_layer = self.e_fc1_methy_18(x[17])
            methy_19_level2_layer = self.e_fc1_methy_19(x[18])
            methy_20_level2_layer = self.e_fc1_methy_20(x[19])
            methy_21_level2_layer = self.e_fc1_methy_21(x[20])
            methy_22_level2_layer = self.e_fc1_methy_22(x[21])
            methy_X_level2_layer = self.e_fc1_methy_X(x[22])

            # concat methy tensor together
            methy_level2_layer = torch.cat((methy_1_level2_layer, methy_2_level2_layer, methy_3_level2_layer,
                                            methy_4_level2_layer, methy_5_level2_layer, methy_6_level2_layer,
                                            methy_7_level2_layer, methy_8_level2_layer, methy_9_level2_layer,
                                            methy_10_level2_layer, methy_11_level2_layer, methy_12_level2_layer,
                                            methy_13_level2_layer, methy_14_level2_layer, methy_15_level2_layer,
                                            methy_16_level2_layer, methy_17_level2_layer, methy_18_level2_layer,
                                            methy_19_level2_layer, methy_20_level2_layer, methy_21_level2_layer,
                                            methy_22_level2_layer, methy_X_level2_layer), 1)

            level_3_layer = self.e_fc2_methy(methy_level2_layer)

            level_4_layer = self.e_fc3(level_3_layer)

            latent_mean = self.e_fc4_mean(level_4_layer)
            latent_log_var = self.e_fc4_log_var(level_4_layer)

            return latent_mean, latent_log_var

        def reparameterize(self, mean, log_var):
            sigma = torch.exp(0.5 * log_var)
            eps = torch.randn_like(sigma)
            return mean + eps * sigma

        def decode(self, z):
            level_4_layer = self.d_fc4(z)

            level_3_layer = self.d_fc3(level_4_layer)
            methy_level3_layer = level_3_layer.narrow(1, 0, level_3_dim_methy)

            methy_level2_layer = self.d_fc2_methy(methy_level3_layer)
            methy_1_level2_layer = methy_level2_layer.narrow(1, 0, level_2_dim_methy)
            methy_2_level2_layer = methy_level2_layer.narrow(1, level_2_dim_methy, level_2_dim_methy)
            methy_3_level2_layer = methy_level2_layer.narrow(1, level_2_dim_methy*2, level_2_dim_methy)
            methy_4_level2_layer = methy_level2_layer.narrow(1, level_2_dim_methy*3, level_2_dim_methy)
            methy_5_level2_layer = methy_level2_layer.narrow(1, level_2_dim_methy*4, level_2_dim_methy)
            methy_6_level2_layer = methy_level2_layer.narrow(1, level_2_dim_methy*5, level_2_dim_methy)
            methy_7_level2_layer = methy_level2_layer.narrow(1, level_2_dim_methy*6, level_2_dim_methy)
            methy_8_level2_layer = methy_level2_layer.narrow(1, level_2_dim_methy*7, level_2_dim_methy)
            methy_9_level2_layer = methy_level2_layer.narrow(1, level_2_dim_methy*8, level_2_dim_methy)
            methy_10_level2_layer = methy_level2_layer.narrow(1, level_2_dim_methy*9, level_2_dim_methy)
            methy_11_level2_layer = methy_level2_layer.narrow(1, level_2_dim_methy*10, level_2_dim_methy)
            methy_12_level2_layer = methy_level2_layer.narrow(1, level_2_dim_methy*11, level_2_dim_methy)
            methy_13_level2_layer = methy_level2_layer.narrow(1, level_2_dim_methy*12, level_2_dim_methy)
            methy_14_level2_layer = methy_level2_layer.narrow(1, level_2_dim_methy*13, level_2_dim_methy)
            methy_15_level2_layer = methy_level2_layer.narrow(1, level_2_dim_methy*14, level_2_dim_methy)
            methy_16_level2_layer = methy_level2_layer.narrow(1, level_2_dim_methy*15, level_2_dim_methy)
            methy_17_level2_layer = methy_level2_layer.narrow(1, level_2_dim_methy*16, level_2_dim_methy)
            methy_18_level2_layer = methy_level2_layer.narrow(1, level_2_dim_methy*17, level_2_dim_methy)
            methy_19_level2_layer = methy_level2_layer.narrow(1, level_2_dim_methy*18, level_2_dim_methy)
            methy_20_level2_layer = methy_level2_layer.narrow(1, level_2_dim_methy*19, level_2_dim_methy)
            methy_21_level2_layer = methy_level2_layer.narrow(1, level_2_dim_methy*20, level_2_dim_methy)
            methy_22_level2_layer = methy_level2_layer.narrow(1, level_2_dim_methy*21, level_2_dim_methy)
            methy_X_level2_layer = methy_level2_layer.narrow(1, level_2_dim_methy*22, level_2_dim_methy)

            recon_x1 = self.d_fc1_methy_1(methy_1_level2_layer)
            recon_x2 = self.d_fc1_methy_2(methy_2_level2_layer)
            recon_x3 = self.d_fc1_methy_3(methy_3_level2_layer)
            recon_x4 = self.d_fc1_methy_4(methy_4_level2_layer)
            recon_x5 = self.d_fc1_methy_5(methy_5_level2_layer)
            recon_x6 = self.d_fc1_methy_6(methy_6_level2_layer)
            recon_x7 = self.d_fc1_methy_7(methy_7_level2_layer)
            recon_x8 = self.d_fc1_methy_8(methy_8_level2_layer)
            recon_x9 = self.d_fc1_methy_9(methy_9_level2_layer)
            recon_x10 = self.d_fc1_methy_10(methy_10_level2_layer)
            recon_x11 = self.d_fc1_methy_11(methy_11_level2_layer)
            recon_x12 = self.d_fc1_methy_12(methy_12_level2_layer)
            recon_x13 = self.d_fc1_methy_13(methy_13_level2_layer)
            recon_x14 = self.d_fc1_methy_14(methy_14_level2_layer)
            recon_x15 = self.d_fc1_methy_15(methy_15_level2_layer)
            recon_x16 = self.d_fc1_methy_16(methy_16_level2_layer)
            recon_x17 = self.d_fc1_methy_17(methy_17_level2_layer)
            recon_x18 = self.d_fc1_methy_18(methy_18_level2_layer)
            recon_x19 = self.d_fc1_methy_19(methy_19_level2_layer)
            recon_x20 = self.d_fc1_methy_20(methy_20_level2_layer)
            recon_x21 = self.d_fc1_methy_21(methy_21_level2_layer)
            recon_x22 = self.d_fc1_methy_22(methy_22_level2_layer)
            recon_x23 = self.d_fc1_methy_X(methy_X_level2_layer)

            return [recon_x1, recon_x2, recon_x3, recon_x4, recon_x5, recon_x6, recon_x7, recon_x8, recon_x9,
                    recon_x10, recon_x11, recon_x12, recon_x13, recon_x14, recon_x15, recon_x16, recon_x17, recon_x18,
                    recon_x19, recon_x20, recon_x21, recon_x22, recon_x23]

        def classifier(self, mean):
            level_1_layer = self.c_fc1(mean)
            level_2_layer = self.c_fc2(level_1_layer)
            output_layer = self.c_fc3(level_2_layer)
            return output_layer

        def forward(self, x):
            mean, log_var = self.encode(x)
            z = self.reparameterize(mean, log_var)
            classifier_x = mean
            if parallel:
                z = z.to('cuda:1')
                classifier_x = classifier_x.to('cuda:1')
            recon_x = self.decode(z)
            pred_y = self.classifier(classifier_x)
            return z, recon_x, mean, log_var, pred_y

    # Instantiate VAE
    if parallel:
        vae_model = VAE()
    else:
        vae_model = VAE().to(device)

    # Early Stopping
    if early_stopping:
        early_stop_ob = Earlystopping()

    # Tensorboard writer
    train_writer = SummaryWriter(log_dir='logs/train')
    val_writer = SummaryWriter(log_dir='logs/val')

    # print the model information
    # print('\nModel information:')
    # print(vae_model)
    total_params = sum(params.numel() for params in vae_model.parameters())
    print('Number of parameters: {}'.format(total_params))

    optimizer = optim.Adam(vae_model.parameters(), lr=learning_rate)

    def methy_recon_loss(recon_x, x):
        loss = F.binary_cross_entropy(recon_x[1], x[1], reduction='sum')
        for i in range(1, 23):
            loss += F.binary_cross_entropy(recon_x[i], x[i], reduction='sum')
        loss /= 23
        return loss

    def kl_loss(mean, log_var):
        loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return loss

    def classifier_loss(pred_y, y):
        loss = F.cross_entropy(pred_y, y, reduction='sum')
        return loss

    # k_methy_recon = 1
    # k_kl = 1
    # k_class = 1

    # loss record
    loss_array = np.zeros(shape=(9, p1_epoch_num+p2_epoch_num+1))
    # performance metrics
    metrics_array = np.zeros(4)

    def train(e_index, e_num, k_methy_recon, k_kl, k_c):
        vae_model.train()
        train_methy_recon = 0
        train_kl = 0
        train_classifier = 0
        train_correct_num = 0
        train_total_loss = 0
        for batch_index, sample in enumerate(train_loader):
            data = sample[0]
            y = sample[1]
            for chr_i in range(23):
                data[chr_i] = data[chr_i].to(device)
            y = y.to(device)
            optimizer.zero_grad()
            _, recon_data, mean, log_var, pred_y = vae_model(data)
            if parallel:
                for chr_i in range(23):
                    recon_data[chr_i] = recon_data[chr_i].to('cuda:0')
                pred_y = pred_y.to('cuda:0')

            methy_recon = methy_recon_loss(recon_data, data)
            kl = kl_loss(mean, log_var)
            class_loss = classifier_loss(pred_y, y)
            loss = k_methy_recon * methy_recon + k_kl * kl + k_c * class_loss

            loss.backward()

            with torch.no_grad():
                pred_y_softmax = F.softmax(pred_y, dim=1)
                _, predicted = torch.max(pred_y_softmax, 1)
                correct = (predicted == y).sum().item()

                train_methy_recon += methy_recon.item()
                train_kl += kl.item()
                train_classifier += class_loss.item()
                train_correct_num += correct
                train_total_loss += loss.item()

            optimizer.step()

            # if batch_index % 15 == 0:
            #     print('Epoch {:3d}/{:3d}  ---  [{:5d}/{:5d}] ({:2d}%)\n'
            #           'Methy Recon Loss: {:.2f}   KL Loss: {:.2f}   '
            #           'Classification Loss: {:.2f}\nACC: {:.2f}%'.format(
            #         e_index + 1, e_num, batch_index * len(data[0]), len(train_dataset),
            #         round(100. * batch_index / len(train_loader)), methy_recon.item() / len(data[0]),
            #         kl.item() / len(data[0]), class_loss.item() / len(data[0]),
            #         correct / len(data[0]) * 100))

        train_methy_recon_ave = train_methy_recon / len(train_dataset)
        train_kl_ave = train_kl / len(train_dataset)
        train_classifier_ave = train_classifier / len(train_dataset)
        train_accuracy = train_correct_num / len(train_dataset) * 100
        train_total_loss_ave = train_total_loss / len(train_dataset)

        print('Epoch {:3d}/{:3d}\n'
              'Training\n'
              'Methy Recon Loss: {:.2f}   KL Loss: {:.2f}   '
              'Classification Loss: {:.2f}\nACC: {:.2f}%'.
              format(e_index + 1, e_num, train_methy_recon_ave, train_kl_ave, train_classifier_ave, train_accuracy))
        loss_array[0, e_index] = train_methy_recon_ave
        loss_array[1, e_index] = train_kl_ave
        loss_array[2, e_index] = train_classifier_ave
        loss_array[3, e_index] = train_accuracy

        # TB
        train_writer.add_scalar('Total loss', train_total_loss_ave, e_index)
        train_writer.add_scalar('Methy recon loss', train_methy_recon_ave, e_index)
        train_writer.add_scalar('KL loss', train_kl_ave, e_index)
        train_writer.add_scalar('Classification loss', train_classifier_ave, e_index)
        train_writer.add_scalar('Accuracy', train_accuracy, e_index)

    if separate_testing:
        def val(e_index, get_metrics=False):
            vae_model.eval()
            val_methy_recon = 0
            val_kl = 0
            val_classifier = 0
            val_correct_num = 0
            val_total_loss = 0
            y_store = torch.tensor([0])
            predicted_store = torch.tensor([0])

            with torch.no_grad():
                for batch_index, sample in enumerate(val_loader):
                    data = sample[0]
                    y = sample[1]
                    for chr_i in range(23):
                        data[chr_i] = data[chr_i].to(device)
                    y = y.to(device)
                    _, recon_data, mean, log_var, pred_y = vae_model(data)
                    if parallel:
                        for chr_i in range(23):
                            recon_data[chr_i] = recon_data[chr_i].to('cuda:0')
                        pred_y = pred_y.to('cuda:0')

                    methy_recon = methy_recon_loss(recon_data, data)
                    kl = kl_loss(mean, log_var)
                    class_loss = classifier_loss(pred_y, y)
                    loss = methy_recon + kl + class_loss

                    pred_y_softmax = F.softmax(pred_y, dim=1)
                    _, predicted = torch.max(pred_y_softmax, 1)
                    correct = (predicted == y).sum().item()

                    y_store = torch.cat((y_store, y.cpu()))
                    predicted_store = torch.cat((predicted_store, predicted.cpu()))

                    val_methy_recon += methy_recon.item()
                    val_kl += kl.item()
                    val_classifier += class_loss.item()
                    val_correct_num += correct
                    val_total_loss += loss.item()

            output_y = y_store[1:].numpy()
            output_pred_y = predicted_store[1:].numpy()

            if get_metrics:
                metrics_array[0] = metrics.accuracy_score(output_y, output_pred_y)
                metrics_array[1] = metrics.precision_score(output_y, output_pred_y, average='weighted')
                metrics_array[2] = metrics.recall_score(output_y, output_pred_y, average='weighted')
                metrics_array[3] = metrics.f1_score(output_y, output_pred_y, average='weighted')

            val_methy_recon_ave = val_methy_recon / len(val_dataset)
            val_kl_ave = val_kl / len(val_dataset)
            val_classifier_ave = val_classifier / len(val_dataset)
            val_accuracy = val_correct_num / len(val_dataset) * 100
            val_total_loss_ave = val_total_loss / len(val_dataset)

            print('Validation\n'
                  'Methy Recon Loss: {:.2f}   KL Loss: {:.2f}   Classification Loss: {:.2f}'
                  '\nACC: {:.2f}%\n'.
                  format(val_methy_recon_ave, val_kl_ave, val_classifier_ave, val_accuracy))
            loss_array[4, e_index] = val_methy_recon_ave
            loss_array[5, e_index] = val_kl_ave
            loss_array[6, e_index] = val_classifier_ave
            loss_array[7, e_index] = val_accuracy

            # TB
            val_writer.add_scalar('Total loss', val_total_loss_ave, e_index)
            val_writer.add_scalar('Methy recon loss', val_methy_recon_ave, e_index)
            val_writer.add_scalar('KL loss', val_kl_ave, e_index)
            val_writer.add_scalar('Classification loss', val_classifier_ave, e_index)
            val_writer.add_scalar('Accuracy', val_accuracy, e_index)

            return val_accuracy, output_pred_y

    print('\nUNSUPERVISED PHASE\n')
    # unsupervised phase
    for epoch_index in range(p1_epoch_num):
        train(e_index=epoch_index, e_num=p1_epoch_num+p2_epoch_num, k_methy_recon=1, k_kl=1, k_c=0)
        if separate_testing:
            _, out_pred_y = val(epoch_index)

    print('\nSUPERVISED PHASE\n')
    # supervised phase
    epoch_number = p1_epoch_num
    for epoch_index in range(p1_epoch_num, p1_epoch_num+p2_epoch_num):
        epoch_number += 1
        train(e_index=epoch_index, e_num=p1_epoch_num+p2_epoch_num, k_methy_recon=0, k_kl=0, k_c=1)
        if separate_testing:
            if epoch_index == p1_epoch_num+p2_epoch_num-1:
                val_classification_acc, out_pred_y = val(epoch_index, get_metrics=True)
            else:
                val_classification_acc, out_pred_y = val(epoch_index)
            if early_stopping:
                early_stop_ob(vae_model, val_classification_acc)
                if early_stop_ob.stop_now:
                    print('Early stopping\n')
                    break

    if early_stopping:
        best_epoch = p1_epoch_num + early_stop_ob.best_epoch_num
        loss_array[8, 0] = best_epoch
        print('Load model of Epoch {:d}'.format(best_epoch))
        vae_model.load_state_dict(torch.load('../ssd/checkpoint.pt'))
        _, out_pred_y = val(epoch_number, get_metrics=True)

    # Encode all of the data into the latent space
    print('Encoding all the data into latent space...')
    vae_model.eval()
    with torch.no_grad():
        d_z_store = torch.zeros(1, latent_dim).to(device)
        for i, sample in enumerate(full_loader):
            d = sample[0]
            for chr_i in range(23):
                d[chr_i] = d[chr_i].to(device)
            _, _, d_z, _, _ = vae_model(d)
            d_z_store = torch.cat((d_z_store, d_z), 0)
    all_data_z = d_z_store[1:]
    all_data_z_np = all_data_z.cpu().numpy()

    # Output file
    print('Preparing the output files... ')
    input_path_name = input_path.split('/')[-1]
    latent_space_path = 'results/' + input_path_name + str(latent_dim) + 'D_latent_space.tsv'

    all_data_z_df = pd.DataFrame(all_data_z_np, index=sample_id)
    all_data_z_df.to_csv(latent_space_path, sep='\t')

    if separate_testing:
        pred_y_path =  'results/' + input_path_name + str(latent_dim) + 'D_pred_y.tsv'
        np.savetxt(pred_y_path, out_pred_y, delimiter='\t')

        metrics_record_path = 'results/' + input_path_name + str(latent_dim) + 'D_metrics.tsv'
        np.savetxt(metrics_record_path, metrics_array, delimiter='\t')

    if output_loss_record:
        loss_record_path = 'results/' + input_path_name + str(latent_dim) + 'D_loss_record.tsv'
        np.savetxt(loss_record_path, loss_array, delimiter='\t')

    return all_data_z_df
