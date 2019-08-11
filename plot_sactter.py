import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_scatter(latent_code, output_path,
                 label_file='data/PANCAN/GDC-PANCAN_both_samples_tumour_type.tsv',
                 colour_file='data/TCGA_colors_obvious.tsv', latent_code_dim=2, have_label=True):
    if latent_code_dim <= 3:
        if latent_code_dim == 3:
            # Plot the 3D scatter graph of latent space
            if have_label:
                # Set sample label
                disease_id = pd.read_csv(label_file, sep='\t', index_col=0)
                latent_code_label = pd.merge(latent_code, disease_id, left_index=True, right_index=True)
                colour_setting = pd.read_csv(colour_file, sep='\t')
                fig = plt.figure(figsize=(8, 5.5))
                ax = fig.add_subplot(111, projection='3d')
                for index in range(len(colour_setting)):
                    code = colour_setting.iloc[index, 1]
                    colour = colour_setting.iloc[index, 0]
                    if code in latent_code_label.iloc[:, latent_code_dim].unique():
                        latent_code_label_part = latent_code_label[latent_code_label.iloc[:, latent_code_dim] == code]
                        ax.scatter(latent_code_label_part.iloc[:, 0], latent_code_label_part.iloc[:, 1],
                                   latent_code_label_part.iloc[:, 2], s=2, marker='o', alpha=0.8, c=colour, label=code)
                ax.legend(ncol=2, markerscale=4, bbox_to_anchor=(1, 0.9), loc='upper left', frameon=False)
            else:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(latent_code.iloc[:, 0], latent_code.iloc[:, 1], latent_code.iloc[:, 2], s=2, marker='o',
                           alpha=0.8)
            ax.set_xlabel('First Latent Dimension')
            ax.set_ylabel('Second Latent Dimension')
            ax.set_zlabel('Third Latent Dimension')
        elif latent_code_dim == 2:
            if have_label:
                # Set sample label
                disease_id = pd.read_csv(label_file, sep='\t', index_col=0)
                latent_code_label = pd.merge(latent_code, disease_id, left_index=True, right_index=True)
                colour_setting = pd.read_csv(colour_file, sep='\t')
                plt.figure(figsize=(8, 5.5))
                for index in range(len(colour_setting)):
                    code = colour_setting.iloc[index, 1]
                    colour = colour_setting.iloc[index, 0]
                    if code in latent_code_label.iloc[:, latent_code_dim].unique():
                        latent_code_label_part = latent_code_label[latent_code_label.iloc[:, latent_code_dim] == code]
                        plt.scatter(latent_code_label_part.iloc[:, 0], latent_code_label_part.iloc[:, 1], s=2,
                                    marker='o', alpha=0.8, c=colour, label=code)
                plt.legend(ncol=2, markerscale=4, bbox_to_anchor=(1, 1), loc='upper left', frameon=False)
            else:
                plt.scatter(latent_code.iloc[:, 0], latent_code.iloc[:, 1], s=2, marker='o', alpha=0.8)
            plt.xlabel('First Latent Dimension')
            plt.ylabel('Second Latent Dimension')
        input_file_name = output_path.split('/')[-1]
        fig_path = 'results/' + input_file_name + str(latent_code_dim) + 'D_fig.png'
        fig_path_svg = 'results/' + input_file_name + str(latent_code_dim) + 'D_fig.svg'
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.savefig(fig_path_svg)
