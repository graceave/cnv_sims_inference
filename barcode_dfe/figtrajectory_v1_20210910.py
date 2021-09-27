#!/usr/bin/python
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.collections import LineCollection
from matplotlib import colors
import matplotlib.ticker as mticker
import argparse


def main():
    """
    ----------------------------------------------------------------------------
    PLOT LINEAGE TRAJECTOTRIES
    ----------------------------------------------------------------------------
    """
    parser = argparse.ArgumentParser(description='Plot lineage trjectories', 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-i', '--input', type=str, required=True, 
                        help='a *.csv file for read number')
    
        
    parser.add_argument('-t', '--time', type=str, required=True,
                        help='a *.csv file: with 1st column being all sequenced '
                             'time-points, and 2nd column being effective cell '
                             'number at each sequenced time-point.')
    
    parser.add_argument('-c', '--color', type=str, default='', 
                        help='a *.csv file for fitness of all lineages')
    
    parser.add_argument('-o', '--output', type=str, default='fig_lineage_trajectory', 
                        help='name of the output figure')
    
    
    args = parser.parse_args()
    
    read_num_seq = np.array(pd.read_csv(args.input, header=None), dtype=float)
    
    tempt = pd.read_csv(args.time, header=None)
    t_seq = np.array(tempt[0], dtype=float)
    cell_depth_seq = np.array(tempt[1], dtype=float)
    
    lineage_num, seq_num = read_num_seq.shape
    if args.color == '':
        x_array = np.zeros(lineage_num)
    else:
        x_array = np.array(pd.read_csv(args.color, header=None)[0], dtype='float') # fitness
   
    file_name = args.output
    
    
    ##### change all 0 cell number in lineages
    pos = np.array([np.max(j) for i in range(lineage_num) for j in [np.where(read_num_seq[i, :] != 0)[0]]]).T
    for i in range(lineage_num):
        pos_tempt = np.where(read_num_seq[i, 0:pos[i]] == 0)[0]
        read_num_seq[i, pos_tempt] = 1

    read_depth_seq = np.sum(read_num_seq, axis=0)
    cell_num_seq = np.round(read_num_seq / read_depth_seq * cell_depth_seq)
    cell_num_seq[cell_num_seq < 1] = 0.1
    
    
    ##### sorting
    data_dict = {k: cell_num_seq[:, k] for k in range(seq_num)}
    data_dict[seq_num] = np.array(x_array, dtype=float)
    data_DataFrame = pd.DataFrame(data_dict)
    data_DataFrame = data_DataFrame.sort_values(by=list(np.flip(data_DataFrame.columns.values)),
                                                    ascending=False)
    idx_sort = data_DataFrame.index
    data_DataFrame = data_DataFrame.reset_index(drop=True)
    tempt = np.array(data_DataFrame, dtype=float)
    cell_num_seq_sort = tempt[:,:-1]
    x_array_sort = tempt[:,-1]
    
 
    ##### random shuffle lineages
    idx_random = np.arange(lineage_num)
    random.shuffle(idx_random)
    segments = [np.column_stack([x, y]) for x, y in zip(np.tile(t_seq, (lineage_num, 1)), 
                                                        cell_num_seq_sort[idx_random, :])]

    
    ##### plotting
    idx_pos = [k for (k,ele) in enumerate(x_array_sort) if ele > 0]
    idx_neu = [k for (k,ele) in enumerate(x_array_sort) if ele == 0]
    idx_neg = [k for (k,ele) in enumerate(x_array_sort) if ele < 0]
    lineage_num_pos, lineage_num_neu, lineage_num_neg = len(idx_pos), len(idx_neu), len(idx_neg)

    fig = plt.figure(figsize=(5.5, 3.5))
    ax = fig.add_axes([0.25, 0.25, 0.65, 0.65])
    #ax.set_facecolor('lightgray')
    
    if lineage_num_pos==lineage_num:
        #print('1')
        color_map = 'Reds'
        color_array = x_array_sort[idx_random]
        lc = LineCollection(segments, array=color_array, cmap=color_map, lw=0.25)

    elif lineage_num_pos>0 and lineage_num_neg>0:
        #print('2')
        color_map = 'RdBu_r'
        color_array = x_array_sort[idx_random]
        divnorm = colors.TwoSlopeNorm(vmin=np.min(color_array), vcenter=0, vmax=np.max(color_array))
        lc = LineCollection(segments, array=color_array, cmap=color_map, norm=divnorm, lw=0.25)
    
    elif lineage_num_pos>0 and lineage_num_neu>0 and lineage_num_neg==0:
        #print('3')        
        idx_random_pos = [k for (k,ele) in enumerate(x_array_sort[idx_random]) if ele > 0]
        idx_random_neu = [k for (k,ele) in enumerate(x_array_sort[idx_random]) if ele == 0]
        segments_neu = [np.column_stack([x, y]) for x, y in zip(np.tile(t_seq, (lineage_num_neu, 1)), 
                                                        cell_num_seq_sort[idx_random[idx_random_neu], :])]
        segments_pos = [np.column_stack([x, y]) for x, y in zip(np.tile(t_seq, (lineage_num_pos, 1)), 
                                                        cell_num_seq_sort[idx_random[idx_random_pos], :])]
        
        color_map = 'Reds'
        color_array = x_array_sort
        color_array_pos = color_array[idx_random[idx_random_pos]]
        singlenorm = colors.Normalize(vmin=0, vmax=np.max(color_array))
        lc = LineCollection(segments_pos, array=color_array_pos, cmap=color_map, norm=singlenorm, lw=0.25)
        #lc = LineCollection(segments_pos, array=color_array_pos, cmap=color_map, lw=0.25)
        lc_neu = LineCollection(segments_neu, color='#007BA7', lw=0.25, alpha=0.05) #blue
        #lc_neu = LineCollection(segments_neu, color='#800080', lw=0.25, alpha=0.05) #purple
        #lc_neu = LineCollection(segments_neu, color='#0504aa', lw=0.25, alpha=0.05) #royal blue
        #lc_neu = LineCollection(segments_neu, color='#0070C0', lw=0.25, alpha=0.05) #blue
        ax.add_collection(lc_neu)

    elif lineage_num_neu==lineage_num:
        #print('4')
        color_map = 'Purples'
        color_array = -np.linspace(1, lineage_num, lineage_num)/lineage_num
        color_array = color_array[idx_random]
        vmin, vmax = np.min(color_array), np.max(color_array)
        lc = LineCollection(segments, array=color_array, cmap=color_map, lw=0.25)

    ax.add_collection(lc)
    plt.yscale('log')
    plt.yticks([10**(3*i-1) for i in range(6)], 
               tuple(['Extinct'] + ['$\\mathdefault{10^{' + str(3*i-1) + '}}$' for i in range(1,6)]))

    x_lim_min, x_lim_max, x_lim_step = t_seq[0], t_seq[-1], (t_seq[-1] - t_seq[0])/20
    y_lim_min, y_lim_max, y_lim_step = -1, np.log10(np.max(cell_num_seq)), (np.log10(np.max(cell_num_seq))-(-1))/20
    plt.xlim(x_lim_min - x_lim_step, x_lim_max + x_lim_step)
    plt.ylim(10**(y_lim_min - y_lim_step), 10**(y_lim_max + y_lim_step*4))

    cbar = fig.colorbar(lc) # color bar setting
    if lineage_num_pos>0 and lineage_num_neg>0:
        cbar.ax.yaxis.set_major_locator(mticker.MaxNLocator(10))
        ticks_loc = cbar.ax.get_yticks().tolist()
        cbar.ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        label_format = '{:,.2f}'
        cbar.ax.set_yticklabels([label_format.format(x) for x in ticks_loc])
    
    elif lineage_num_pos>0 and lineage_num_neu>0 and lineage_num_neg==0:
        cbar.ax.yaxis.set_major_locator(mticker.MaxNLocator(10))
        ticks_loc = cbar.ax.get_yticks().tolist()
        cbar.ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        label_format_1 = '{:,.0f}' + '%'
        label_format_2 = '{:,.2f}'
        tempt = []
        for x in ticks_loc:
            if x<0:
                tempt.append(label_format_1.format(100*abs(x)/np.max(color_array)))
            else:
                tempt.append(label_format_2.format(x))
        cbar.ax.set_yticklabels(tempt)
    
    elif lineage_num_neu==lineage_num:
        cbar.ax.yaxis.set_major_locator(mticker.MaxNLocator(10))
        ticks_loc = cbar.ax.get_yticks().tolist()
        cbar.ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        label_format = '{:,.0f}' + '%'
        cbar.ax.set_yticklabels([label_format.format(100*abs(x)) for x in ticks_loc])

    fig.text(0.515, 0.12, 'Time (generations)', ha='center', va='center', fontsize=12)
    fig.text(0.125, 0.6, 'Cell number', ha='center', va='center', rotation='vertical', fontsize=12)
    #plt.show()
    plt.savefig(file_name + '.pdf', transparent=True)
    #plt.savefig('./fig_trajectory_' + file_name + '.pdf')
        
if __name__ == "__main__":
    main()
    
