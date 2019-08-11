# DNA methylation

import numpy as np
import pandas as pd

# Select both samples
both_ids = np.loadtxt('data/PANCAN/GDC-PANCAN_both_samples.tsv', delimiter='\t', dtype='U32')
both_ids_index = np.insert(both_ids, 0, 'Composite Element REF')

file_path = 'data/PANCAN/GDC-PANCAN_methylation450_preprocessed.tsv'
# DNA methylation: 392761 rows × 8764 columns
input_df = pd.read_csv(file_path, sep='\t', header=0, index_col=0, usecols=both_ids_index)[both_ids]

# Select specific chr
all_index_set = set(input_df.index)

mapping = pd.read_csv('data/illuminaMethyl450_hg38_GDC', sep='\t', header=0, index_col=0)

chrs = mapping['chrom'].unique()
chrs = np.delete(chrs, 17)

# Store the number of probes for each chromosome
chrs_number_dict = {'chrs':list(chrs), 'in_mapping':list(np.zeros(24)), 'in_data':list(np.zeros(24))}
chrs_number_df = pd.DataFrame(chrs_number_dict)
chrs_number_df.set_index(['chrs'], inplace=True)

for chrom in chrs:
    chr_index_set = set(mapping[mapping['chrom'] == chrom].index)
    chrs_number_df.loc[chrom, 'in_mapping'] = len(chr_index_set)
    chr_index_exi_set = all_index_set & chr_index_set
    chrs_number_df.loc[chrom, 'in_data'] = len(chr_index_exi_set)
    chr_index_exi_array = np.array(list(chr_index_exi_set))

    chr_df = input_df.loc[chr_index_exi_array]
    output_path = 'data/PANCAN/GDC-PANCAN_methylation450_preprocessed_both_' + chrom + '.tsv'
    chr_df.to_csv(output_path, sep='\t')

chrs_number_df.to_csv('data/PANCAN/GDC-PANCAN_methylation450_preprocessed_chr_number.tsv', sep='\t')
input_df.to_csv('data/PANCAN/GDC-PANCAN_methylation450_preprocessed_both.tsv', sep='\t')


# Combine methy and expr data to a single file

# P
input_path = 'data/PANCAN/GDC-PANCAN_'

sample_id = np.loadtxt(input_path + 'both_samples.tsv', delimiter='\t', dtype='str')

expr_path = input_path + 'htseq_fpkm_'
methy_path = input_path + 'methylation450_'

# Set the dtype to f32 for memory saving purpose
all_cols_f32 = {col: np.float32 for col in sample_id}

print('Loading gene expression data...')
expr_df = pd.read_csv(expr_path + 'preprocessed_both.tsv', sep='\t', header=0, index_col=0, dtype=all_cols_f32)

print('Loading DNA methylation data...')
methy_df = pd.read_csv(methy_path + 'preprocessed_both.tsv', sep='\t', header=0, index_col=0, dtype=all_cols_f32)

multi_df = pd.concat([methy_df, expr_df])
out_path = input_path + 'preprocessed_both.tsv'
multi_df.to_csv(out_path, sep='\t')
