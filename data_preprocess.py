import numpy as np
import pandas as pd


def data_preprocess(input_path, file_delimiter='\t', file_header=None, file_index_col=None, file_transpose=False,
                    norm_0_1=True, probe_filter=None, statistic_select_num=0, statistic_metrics='std'):
    print('Loading the input data...')
    full_input_path = '../data/' + input_path
    # use float32 to save memory
    data_df_test = pd.read_csv(full_input_path, sep=file_delimiter, header=file_header, index_col=file_index_col,
                               nrows=10)
    cols_f32 = {col: np.float32 for col in data_df_test}
    data_df = pd.read_csv(full_input_path, sep=file_delimiter, header=file_header, index_col=file_index_col,
                          dtype=cols_f32)

    print('Pre-processing the input data...')
    # Delete selected probes
    if probe_filter:
        filter_list = np.loadtxt(probe_filter, delimiter='\t', dtype=str)
        if file_transpose:
            data_df = data_df.drop(filter_list)
        else:
            data_df = data_df.drop(filter_list, axis=1)

    # Deal with nan value
    if file_transpose:
        data_df.dropna(axis=0, thresh=data_df.shape[1] * 0.9, inplace=True)
        data_df.dropna(axis=1, thresh=data_df.shape[0] * 0.9, inplace=True)
    else:
        data_df.dropna(axis=1, thresh=data_df.shape[0] * 0.9, inplace=True)
        data_df.dropna(axix=0, thresh=data_df.shape[1] * 0.9, inplace=True)

    # Use feature average to fill na
    if file_transpose:
        row_mean = data_df.mean(axis=1)
        for col_index, col_name in enumerate(data_df):
            data_df.iloc[:, col_index].fillna(row_mean, inplace=True)
    else:
        data_df.fillna(data_df.mean(axis=0), inplace=True)

    # Normalize the dataframe to the range of 0-1
    if norm_0_1:
        # Min-max normalization
        data_df = (data_df - data_df.min().min()) / (data_df.max().max() - data_df.min().min())

    input_path_name = input_path.split('.')[0]

    # Select certain number of probes according to some statistic metrics
    if statistic_select_num > 0:
        if statistic_metrics == 'mad':
            if file_transpose:
                select_index = data_df.mad(axis=1).sort_values(ascending=False)[:statistic_select_num].index
                data_df = data_df.loc[select_index]
            else:
                select_column = data_df.mad(axis=0).sort_values(ascending=False)[:statistic_select_num].index
                data_df = data_df.loc[:, select_column]
        else:
            if file_transpose:
                select_index = data_df.std(axis=1).sort_values(ascending=False)[:statistic_select_num].index
                data_df = data_df.loc[select_index]
            else:
                select_column = data_df.std(axis=0).sort_values(ascending=False)[:statistic_select_num].index
                data_df = data_df.loc[:, select_column]
        output_path = '../data/' + input_path_name + '_' + str(statistic_select_num) + '_' + statistic_metrics + '.tsv'
        data_df.to_csv(output_path, sep='\t')
    else:
        output_path = '../data/' + input_path_name + '_preprocessed.tsv'
        data_df.to_csv(output_path, sep='\t')

    return data_df
