# stdlib
import os
from os.path import join
import re

# external
from PIL import Image
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# local
# import config
from dataset_rose_youtu import read_annotations, labels_orig


def plot_category_counts_per_id(paths_all):
    """ Plot category counts per ID as multiple figures """
    for id0 in np.unique(paths_all['id0']):
        sns.histplot(paths_all[paths_all['id0'] == id0]['label_num'])  # changed 'label_num' to 'label_unif' on Mar 21
        plt.title(f'Label distribution for ID {id0}')
        plt.show()


def plot_category_counts_all_ids(paths_all):
    """ Plot category counts per ID in one figure """
    fig, axs = plt.subplots(2, 5, figsize=(20, 10))
    fig.suptitle('Label distribution for all IDs', fontsize=16)
    for i, id0 in enumerate(np.unique(paths_all['id0'])):
        sns.histplot(paths_all[paths_all['id0'] == id0]['label_num'],
                     ax=axs[i // 5, i % 5])  # changed 'label_num' to 'label_unif' on Mar 21
        axs[i // 5, i % 5].set_title(f'Label distribution for ID {id0}')
        axs[i // 5, i % 5].set_xlabel('Label')
        # set axis ticks to be integers
        axs[i // 5, i % 5].set_xticks(range(0, 8))
    plt.tight_layout()
    plt.show()


def plot_example_images(paths_all):
    sns.set_style('whitegrid')
    sns.set_context('paper', font_scale=2)
    sns.set_palette('colorblind')  # does nothing
    fig, axs = plt.subplots(2, 4, figsize=(14, 9))
    for i, label_num in enumerate(labels_orig.values()):
        # get first image for each label
        try:
            img_path = paths_all[paths_all['label_num'] == i]['path'].iloc[
                0]  # changed 'label_num' to 'label_unif' on Mar 21
        except:
            print('No images for label', label_num)
            img = np.zeros((256, 256, 3))

        img = Image.open(img_path)
        plt.subplot(2, 4, i + 1)
        plt.imshow(img)
        plt.title(label_num)
        plt.axis('off')

    fig.suptitle('RoseYoutu Example Images', fontsize=32)
    plt.tight_layout()

    # zero padding around figure
    plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01)

    path = join('images_plots', 'rose_youtu_example_images.pdf')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)

    plt.show()


# if __name__ == '__main__':
def main():
    """No docstring"""
    ''' Load data annotations '''
    paths_genuine = read_annotations('genuine')
    paths_attacks = read_annotations('attack')
    paths_all = pd.concat([paths_genuine, paths_attacks])

    ''' Plots styling '''
    sns.set_style('whitegrid')
    sns.set_context('paper', font_scale=1.5)
    sns.set_palette('colorblind')  # does nothing

    ''' Plotting '''
    if False:
        ''' Label distribution per ID '''
        # Unused, info shown in a single plot below
        # plot_category_counts_per_id(paths_all)

        plot_category_counts_all_ids(paths_all)

        ''' Explore data '''
        # pandas dataframe - index by id0, columns by label_num
        df = paths_all[['id0', 'label_num']].groupby('id0').value_counts().unstack(
            fill_value=0)  # changed 'label_num' to 'label_unif' on Mar 21
        print(df)

        # unique tuples of (id0, label_num)
        unique_id0_label_num, counts = np.unique(paths_all[['id0', 'label_num']].to_numpy(), axis=0,
                                                 return_counts=True)  # changed 'label_num' to 'label_unif' on Mar 21

        ''' Unique values per column '''
        print('Unique values per column')
        for col in paths_all.columns:
            uniq_vals, counts = np.unique(paths_all[col], return_counts=True)
            # pad string to 20 chars
            print(f'{col:15} {len(uniq_vals):6}')

    ''' Plot example images '''
    plot_example_images(paths_all)

    if False:
        ''' Plot attack category distribution (paths_all) '''
        paths_all[
            'label_num'].value_counts().plot.bar()  # changed 'label_num' to 'label_unif' on Mar 21'].value_counts().plot.bar()
        plt.title('Attack Category distribution')
        plt.xlabel('Attack Category')
        plt.ylabel('Count')
        plt.xticks(np.arange(len(cat_names)), cat_names, rotation=45)
        plt.tight_layout()
        plt.show()

    if False:
        ''' Plot categories per person ID, per dataset subset '''
        for subset in [paths_genuine]:  # '[paths_train, paths_val, paths_test, paths_all]:
            # get variable name as string
            subset_name = [k for k, v in locals().items() if v is subset][0]

            df_train = subset[['id0', 'label_num']].groupby('id0').value_counts().unstack(fill_value=0).plot.bar(
                # changed 'label_num' to 'label_unif' on Mar 21']].groupby('id0').value_counts().unstack(fill_value=0).plot.bar(
                stacked=True)
            # name legend by categories_names (not x-axis)
            df_train.legend(cat_names, loc='center left', bbox_to_anchor=(1, 0.5))

            # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.title(f'Attack Category distribution per person ID ({subset_name})')
            plt.xlabel('Person ID (id0)')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.show()
