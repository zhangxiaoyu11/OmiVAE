import numpy as np
import pandas as pd
from MultiOmiVAE import MultiOmiVAE
from MethyOmiVAE import MethyOmiVAE
from ExprOmiVAE import ExprOmiVAE
from plot_sactter import plot_scatter
from classification import classification


if __name__ == "__main__":
    input_path = 'data/OmiVAE/PANCAN/GDC-PANCAN_'

    expr_path = input_path + 'htseq_fpkm_'
    methy_path = input_path + 'methylation450_'

    # Loading data

    print('Loading gene expression data...')
    expr_df = pd.read_csv(expr_path + 'preprocessed_both.tsv', sep='\t', header=0, index_col=0)

    print('Loading DNA methylation data...')
    methy_chr_df_list = []
    chr_id = list(range(1, 23))
    chr_id.append('X')
    # Loop among different chromosomes
    for chrom in chr_id:
        print('Loading methylation data on chromosome ' + str(chrom) + '...')
        methy_chr_path = methy_path + 'preprocessed_both_chr' + str(chrom) + '.tsv'
        # methy_chr_df = pd.read_csv(methy_chr_path, sep='\t', header=0, index_col=0, dtype=all_cols_f32)
        methy_chr_df = pd.read_csv(methy_chr_path, sep='\t', header=0, index_col=0)
        methy_chr_df_list.append(methy_chr_df)

    e_num_1 = 50
    e_num_2 = 200
    l_dim = 128

    # Example
    latent_code, train_acc, val_acc = MultiOmiVAE(input_path=input_path, expr_df=expr_df,
                                                  methy_chr_df_list=methy_chr_df_list, p1_epoch_num=e_num_1,
                                                  p2_epoch_num=e_num_2, latent_dim=l_dim, early_stopping=False)