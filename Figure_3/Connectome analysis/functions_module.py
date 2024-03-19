from neuprint import Client
token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImFsZXgubWF1c3NAY2FudGFiLm5ldCIsImxldmVsIjoibm9hdXRoIiwiaW1hZ2UtdXJsIjoiaHR0cHM6Ly9saDMuZ29vZ2xldXNlcmNvbnRlbnQuY29tL2EtL0FBdUU3bUJETVJtWTZGb3NlczZCWkZ5dU40TmFqMDBKeFZ5eWpOR1pQck5fP3N6PTUwP3N6PTUwIiwiZXhwIjoxNzYwODg4NjgyfQ.jT885mSKrCEup0koFvv4-daJgen6WriZ33lw-3R0V8w'
c = Client('neuprint.janelia.org', dataset='hemibrain:v1.0.1', token=token)
c.fetch_version()

from neuprint import fetch_custom_neurons
from neuprint import fetch_shortest_paths
from neuprint import fetch_simple_connections
from neuprint import SegmentCriteria
from neuprint import fetch_shortest_paths
from neuprint import fetch_synapses

import os.path
import sys
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from matplotlib import transforms
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from scipy.linalg import norm


def remove_unnamed_col(df):
    df = df[df.filter(regex='^(?!Unnamed)').columns]
    return df


def load_cell_types(file_path):
    parent_folder = 'neuron_bodyIds_names/'

    cell_type_df = remove_unnamed_col( pd.read_excel( (parent_folder + file_path) ) )
    cell_type_names = list( cell_type_df.name.values )
    cell_type_bodyIds = list( cell_type_df.bodyId.values )

    return cell_type_df, cell_type_names, cell_type_bodyIds


def get_connectivity_matrix_LOP(data_pre, description_pre, data_post, description_post, **kwargs):
    """Function takes bodyIds and names of presynaptic and postsynaptic cells as argument and returns a dictionary containing this information as well as a list with all connectivity results and a connectivity matrix between all pre- and postsynaptic elements.
    data_pre: list of lists. Each list corresponds to one group of cells and contains their bodyIds.
    description_pre: list of strings describing each cell group.
    IMPORTANT: connectivity is restricted to Lobula Plate!
    """
    kwarguments = {'min_weight': 1}
    kwarguments.update(kwargs)
    min_weight = kwarguments['min_weight']

    connectivity_matrix = np.zeros((len(data_pre), len(data_post)))

    num_connections = len(data_pre) * len(data_post)

    results_list = []

    count = 0
    for i, data_pre_i in enumerate(data_pre):
        for j, data_post_j in enumerate(data_post):
            count = count+1
            downstream_criteria = SegmentCriteria(bodyId=data_post_j)
            upstream_criteria = SegmentCriteria(bodyId=data_pre_i)
            results = fetch_simple_connections(upstream_criteria=upstream_criteria, downstream_criteria=downstream_criteria, min_weight=min_weight)

            conn_roiInfo_list = list( results.conn_roiInfo.values )
            LOP_weights = []
            LO_weights = []
            for conn_roiInfo_i in conn_roiInfo_list:
                if "LOP(R)" in conn_roiInfo_i.keys():
                    LOP_conn = conn_roiInfo_i["LOP(R)"]
                    LOP_weights.append( LOP_conn["pre"] )
                else:
                    LOP_weights.append( 0 )
                if "LO(R)" in conn_roiInfo_i.keys():
                    LO_conn = conn_roiInfo_i["LO(R)"]
                    LO_weights.append( LO_conn["pre"] )
                else:
                    LO_weights.append( 0 )

            results["LOP_weights"] = LOP_weights
            results["LO_weights"] = LO_weights
            results_list.append( results )
            connectivity_matrix[i,j] = results.LOP_weights.sum() # only take weights from Lobula Plate into account

#            print(str(round(count/num_connections*100)), ' percent complete', end='\r')
            progress(count, num_connections, 'fetch connectivity')

    connectivity_dict = {}
    connectivity_dict['connectivity_matrix'] = connectivity_matrix
    connectivity_dict['description_pre'] = description_pre
    connectivity_dict['description_post'] = description_post
    connectivity_dict['data_pre'] = data_pre
    connectivity_dict['data_post'] = data_post
    connectivity_dict['results_list'] = results_list
    return connectivity_dict


def plot_connectivity_matrix(connectivity_dict, **kwargs):
    connectivity_matrix = connectivity_dict['connectivity_matrix']
    description_pre = connectivity_dict['description_pre']
    description_post = connectivity_dict['description_post']

    if len(description_pre) > 8 & len(description_post) > 8:
        fig_width = len(description_post)/2.0
        fig_height = len(description_pre)/2.0
    elif len(description_pre) < 5 or len(description_post) < 5:
        fig_width = 5
        fig_height = 5
    else:
        fig_width = len(description_post)/1.5
        fig_height = len(description_pre)/1.5

    kwarguments = {'norm_per_row': True,
               'norm_per_col': False,
               'mask_entries': None,
               'fig_width': fig_width,
               'fig_height': fig_height,
               'font_size': 8,
               'line_width': 1,
               'fill_quadrants': [[],],
               'write_0': True,
               'full_matrix': True}
    kwarguments.update(kwargs)

    write_0 = kwarguments['write_0']
    fill_quadrants = kwarguments['fill_quadrants']
    font_size = kwarguments['font_size']
    line_width = kwarguments['line_width']
    fig_width = kwarguments['fig_width']
    fig_height = kwarguments['fig_height']
    full_matrix = kwarguments['full_matrix']
    norm_per_row = kwarguments['norm_per_row']
    norm_per_col = kwarguments['norm_per_col']
    if norm_per_col:
        norm_per_row = False
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fig_width,fig_height))
    
    len_pre = np.size(connectivity_matrix,0)
    len_post = np.size(connectivity_matrix,1)

    if norm_per_col:
        connectivity_matrix_norm = np.copy(connectivity_matrix)
        for j in (range(len_post)):
            max_val = np.max(connectivity_matrix[:, j])
            if max_val > 0.0:
                connectivity_matrix_norm[:, j] = np.copy(connectivity_matrix[:, j])/max_val
                
    elif norm_per_row:
        connectivity_matrix_norm = np.copy(connectivity_matrix)
        for i in (range(len_pre)):
            max_val = np.max(connectivity_matrix[i, :])
            if max_val > 0.0:
                connectivity_matrix_norm[i, :] = np.copy(connectivity_matrix[i, :])/max_val
    else:
        max_val = np.max(connectivity_matrix)
        connectivity_matrix_norm = connectivity_matrix / max_val

    for i in range(len_pre):
        for j in range(len_post):
            for fill_quadrant in fill_quadrants:
                if i in fill_quadrant and j in fill_quadrant:
                    ax.fill_between((j,j+1), i, i+1, color='k', alpha=0.1)
            ax.fill_between((j,j+1), i, i+1, color='r', alpha=connectivity_matrix_norm[i, j])
            ax.plot((0, len_post+1),(i, i), color='k', alpha=0.2, linewidth=line_width)
            ax.plot((j, j),(0, len_pre+1), color='k', alpha=0.2, linewidth=line_width)
            
            weight = connectivity_matrix[i, j]
            if weight > 0 or write_0:
                ax.text(j+0.5, i+0.5, str( int(weight) ), color='k',
                       ha='center',va='center', fontsize=font_size)

    # Calculate sum for each row and column
    col_sum = np.sum(connectivity_matrix, axis=0)
    col_sum_norm = col_sum/np.max(col_sum)
    row_sum = np.sum(connectivity_matrix, axis=1)
    row_sum_norm = row_sum/np.max(row_sum)
    
    # Fields for row and col sum (total output/input weight for each cell)
    for i in range(len_pre):
        for j in range(len_post, len_post+1):
            weight = row_sum[i]
            weight_norm = row_sum_norm[i]
            ax.fill_between((j,j+1), i, i+1, color='c', alpha=weight_norm)
            ax.plot((0, len_post+1),(i, i), color='k', alpha=0.2, linewidth=line_width)
            ax.plot((j, j),(0, len_pre+1), color='k', alpha=0.2, linewidth=line_width)
        
            if weight > 0:
                ax.text(j+0.5, i+0.5, str( int(weight) ), color='k',
                       ha='center',va='center', fontsize=font_size)
    
    for i in range(len_pre, len_pre+1):
        for j in range(len_post):
            weight = col_sum[j]
            weight_norm = col_sum_norm[j]
            ax.fill_between((j,j+1), i, i+1, color='c', alpha=weight_norm)
            ax.plot((0, len_post+1),(i, i), color='k', alpha=0.2, linewidth=line_width)
            ax.plot((j, j),(0, len_pre+1), color='k', alpha=0.2, linewidth=line_width)
        
            if weight > 0:
                ax.text(j+0.5, i+0.5, str( int(weight) ), color='k',
                       ha='center',va='center', fontsize=font_size)
  
#     if  kwarguments['mask_entries'] is not None:
#         mask_entries = kwarguments['mask_entries']
#         for tup in mask_entries:
#             i, j = tup
#             ax.fill_between((i,i+1), j, j+1, color='k', alpha=1.0)

    ax.set_ylabel('presynaptic')
    ax.set_xlabel('postsynaptic')
    
    tick_positions = np.linspace(0.5,len_post+0.5,len_post+1)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(description_post + ['sum'], rotation=45, ha='right', fontsize=font_size)
    
    tick_positions = np.linspace(0.5,len_pre+0.5,len_pre+1)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(description_pre + ['sum'], fontsize=font_size)
    
    ax.axis([0, len_post+1, 0, len_pre+1])
    ax.set_aspect('equal')
    fig.tight_layout()

    return fig


def check_proportional_input(LP_cells_ids_labels, cell_type_post, **kwargs):
    kwarguments = {'restrict_to_LOP': False}
    kwarguments.update(kwargs)
    restrict_to_LOP = kwarguments['restrict_to_LOP']
    if restrict_to_LOP:
        print('Analysis restricted to Lobula Plate!')
        print('')

    # Fetch (presynaptic) connections
    downstream_criteria = SegmentCriteria(bodyId = list( LP_cells_ids_labels[LP_cells_ids_labels.type==cell_type_post].bodyId.values ) )
    upstream_criteria = SegmentCriteria()
    results = fetch_simple_connections(upstream_criteria=upstream_criteria, downstream_criteria=downstream_criteria, min_weight=1)
 
    # get weights for Lobula and Lobula Plate individually
    conn_roiInfo_list = list( results.conn_roiInfo.values )
    LOP_weights = []
    LO_weights = []
    for conn_roiInfo_i in conn_roiInfo_list:
        if "LOP(R)" in conn_roiInfo_i.keys():
            LOP_conn = conn_roiInfo_i["LOP(R)"]
            if "pre" in LOP_conn.keys():
                LOP_weights.append( LOP_conn["pre"] )
            else:
                LOP_weights.append( 0 )
        else:
            LOP_weights.append( 0 )
        if "LO(R)" in conn_roiInfo_i.keys():
            LO_conn = conn_roiInfo_i["LO(R)"]
            if "pre" in LO_conn.keys():
                LO_weights.append( LO_conn["pre"] )
            else:
                LO_weights.append( 0 )
        else:
            LO_weights.append( 0 )
    results["LOP_weight"] = LOP_weights
    results["LO_weight"] = LO_weights
 
    # Assign cell types
    bodyId_post_unique = results.bodyId_post.unique()
    name_post_list = []
    for bodyId_post_i in bodyId_post_unique:
        name_post = LP_cells_ids_labels[LP_cells_ids_labels.bodyId==bodyId_post_i].name.values[0]
        name_post_list.append( name_post )
        type_post = LP_cells_ids_labels[LP_cells_ids_labels.bodyId==bodyId_post_i].type.values[0]

        results_i = results[results.bodyId_post==bodyId_post_i]
        results_i = results_i.assign(type_post=type_post)
        
        bodyId_i_pre_unique = results_i.bodyId_pre.unique()
        for bodyId_pre_j in bodyId_i_pre_unique:
            if bodyId_pre_j in list( LP_cells_ids_labels.bodyId.values ):
                name_pre = LP_cells_ids_labels[LP_cells_ids_labels.bodyId==bodyId_pre_j].name.values[0]
                type_pre = LP_cells_ids_labels[LP_cells_ids_labels.bodyId==bodyId_pre_j].type.values[0]
            else:
                name_pre = 'unknown'
                type_pre = 'unknown'

            results_i.loc[results_i.bodyId_pre==bodyId_pre_j, 'type_pre'] = type_pre

        results[results.bodyId_post==bodyId_post_i] = results_i
        
    print('postsynaptic cell type: {}'.format(cell_type_post))
    print('number of postsynaptic cells: {}'.format(bodyId_post_unique.size))
    print('number of connections found: {}'.format(results.shape[0]))
    print('number of unique presynaptic cells: {}'.format(results.bodyId_pre.unique().size))
    print('')
    
    def print_inputs(input_type):
        print('number of unique presynaptic {} cells: {}'.format(input_type, results[results.type_pre==input_type].bodyId_pre.unique().size))
        
    print_inputs('T4c')
    print_inputs('T5c')
    print_inputs('T4d')
    print_inputs('T5d')
    print_inputs('VS')
    print_inputs('layer3_LPTC')
    print_inputs('unknown')

    print('')
    print('input weight grouped by presynaptic type')
    # restrict contacts to lobula plate (choose 'weight' instead of 'LOP_weight' if all synaptic contacts are taken into account)
    if restrict_to_LOP:
        results_grouped = results[['type_pre','LOP_weight']].groupby('type_pre').sum()
    else:
        results_grouped = results[['type_pre','weight']].groupby('type_pre').sum()
        
    results_grouped.apply(print)
    
    # for each postsynaptic cell, group input weight by presynaptic type
    individual_post_cell_grouped = []
    individual_post_cell_bodyId = []
    for bodyId_post_i in bodyId_post_unique:
        results_i = results[results.bodyId_post==bodyId_post_i]
        # restrict contacts to lobula plate (choose 'weight' instead of 'LOP_weight' if all synaptic contacts are taken into account)
        if restrict_to_LOP:
            individual_post_cell_grouped.append( results_i[['type_pre','LOP_weight']].groupby('type_pre').sum() )
        else:
            individual_post_cell_grouped.append( results_i[['type_pre','weight']].groupby('type_pre').sum() )
        individual_post_cell_bodyId.append( bodyId_post_i )

    output_dict = {}
    output_dict["results"] = results
    output_dict["results_grouped"] = results_grouped
    output_dict["individual_post_cell_grouped"] = individual_post_cell_grouped
    output_dict["individual_post_cell_bodyId"] = individual_post_cell_bodyId
    output_dict["cell_type_post"] = cell_type_post
    output_dict["name_post_list"] = name_post_list
    
    return output_dict
    
    
def fetch_skeletons_and_synapses(bodyIds, **kwargs):
    """Function loads skeleton and synapse files (optional) from disk, if available. If not, downloads files from neuprint. Furthermore, convex hull as a measure of neuron shape is computed from synapse coordinates in 2D (also optional). """
    kwarguments = {'compute_convex_hull': False,
                   'max_num': len(bodyIds),
                   'synapses': False,
                   'min_confidence': 0.0,
                   'names': [],
                   'print_progress': True}
    kwarguments.update(kwargs)

    max_num = kwarguments['max_num']
    min_confidence = kwarguments['min_confidence']

    skeletons = []
    synapses = []
    hulls_synapses = []

    for i, bodyId in enumerate(bodyIds):
        if kwarguments['print_progress']:
            print(bodyId)
        fname_skel = 'skeleton_files/' + str(bodyId) + '.csv'
        fname_synapse = 'synapse_files/' + str(bodyId) + '.csv'
        if i >= max_num:
            break

        if os.path.isfile(fname_skel):
            skeleton_i = pd.read_csv(fname_skel)
            if kwarguments['print_progress']:
                print('skeleton file read from disk')
        else:
            skeleton_i = c.fetch_skeleton(bodyId, heal=True, format='pandas')
            skeleton_i.to_csv(fname_skel)
            if kwarguments['print_progress']:
                print('skeleton file imported from neuprint server')
        skeletons.append( skeleton_i )
        
        if kwarguments['synapses']:
            if os.path.isfile(fname_synapse):
                synapses_i = pd.read_csv(fname_synapse)
                if kwarguments['print_progress']:
                    print('synapse file read from disk')
            else:
                synapses_i = fetch_synapses(SegmentCriteria(bodyId=bodyId))
                synapses_i.to_csv(fname_synapse)
                if kwarguments['print_progress']:
                    print('synapse file imported from neuprint server')
                
            synapses_i = synapses_i[synapses_i.confidence>min_confidence]
            synapses.append( synapses_i )
            if kwarguments['compute_convex_hull']:
                syn_x_z = np.column_stack((synapses_i.x.values, synapses_i.z.values))
                hulls_synapses.append( ConvexHull(syn_x_z) )

        if kwarguments['print_progress']:
            print('')
    skeletons_bodyids = {}
    skeletons_bodyids["skeletons"] = skeletons
    skeletons_bodyids["bodyIds"] = bodyIds
    skeletons_bodyids["synapses"] = synapses
    skeletons_bodyids["hulls_synapses"] = hulls_synapses
    skeletons_bodyids["names"] = kwarguments['names']

    print('Done')
    return skeletons_bodyids
    
    
def plot_each_neuron_individual(skeletons_bodyids, **kwargs):
    bodyIds = skeletons_bodyids["bodyIds"]
    skeletons = skeletons_bodyids["skeletons"]
    synapses = skeletons_bodyids["synapses"]
    hulls_synapses = skeletons_bodyids["hulls_synapses"]

    kwarguments = {'axis_on': True,
                   'plot_skels': True,
                   'plot_synapses': True,
                   'plot_hull_synapses': True,
                   'nrows': 3,
                   'ncols': 4,
                   'min_confidence_pre': 0.6,
                   'min_confidence_post': 0.6}
    kwarguments.update(kwargs)
    min_conf_pre = kwarguments["min_confidence_pre"]
    min_conf_post = kwarguments["min_confidence_post"]
    
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12,10))

    for i, ax in enumerate(fig.axes):
        if i >= len(skeletons_bodyids["skeletons"]):
            ax.axis('off')
        else:
            print(bodyIds[i])
            if kwarguments['plot_skels']:
                skeleton_i = skeletons[i]
                ax.plot(skeleton_i.x, skeleton_i.z, 'o', color='k', markersize=1, alpha=0.1)

            if kwarguments['plot_synapses']:
                synapses_i = synapses[i]
                synapses_pre_i = synapses_i[(synapses_i.type=='pre') & (synapses_i.confidence>min_conf_pre)]
                synapses_post_i = synapses_i[(synapses_i.type=='post') & (synapses_i.confidence>min_conf_post)]
#                 ax.scatter(synapses_i.x, synapses_i.z, s=1, c='c', alpha=0.5)
                ax.scatter(synapses_pre_i.x, synapses_pre_i.z, s=2, c='r', alpha=0.5)
                ax.scatter(synapses_post_i.x, synapses_post_i.z, s=2, c='c', alpha=0.5)
                
            if kwarguments['plot_hull_synapses']:
                synapses_i = synapses[i]
                hulls_synapses_i = hulls_synapses[i]
                ax.plot(synapses_i.x.values[hulls_synapses_i.vertices], synapses_i.z.values[hulls_synapses_i.vertices], 'm--')
                x_first = synapses_i.x.values[hulls_synapses_i.vertices[0]]
                x_last = synapses_i.x.values[hulls_synapses_i.vertices[-1]]
                z_first = synapses_i.z.values[hulls_synapses_i.vertices[0]]
                z_last = synapses_i.z.values[hulls_synapses_i.vertices[-1]]
                ax.plot([x_last, x_first], [z_last, z_first], 'm--')

            ax.set_title(str(bodyIds[i]))
            ax.set_aspect('equal')
            ax.axis([0, 6000, 20000, 37000])

            if kwarguments['axis_on'] == False:
                ax.axis('off')


def gen_random_colors(num):
    colors = []
    for i in range(num):
        colors.append(list(np.random.choice(range(10), size=3)/10))
    return colors


def plot_each_neuron_together(skeletons_bodyids, **kwargs):
    bodyIds = skeletons_bodyids["bodyIds"]
    skeletons = skeletons_bodyids["skeletons"]
    synapses = skeletons_bodyids["synapses"]
    hulls_synapses = skeletons_bodyids["hulls_synapses"]

    kwarguments = {'axis_on': True,
                   'plot_skels': True,
                   'plot_synapses': True,
                   'plot_hull_synapses': True,
                   'min_confidence_pre': 0.6,
                   'min_confidence_post': 0.6}
    kwarguments.update(kwargs)
    min_conf_pre = kwarguments["min_confidence_pre"]
    min_conf_post = kwarguments["min_confidence_post"]
    
    rand_cols = gen_random_colors(len(bodyIds))
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,10))

    for i, bodyId in enumerate(bodyIds):
        print(bodyIds[i])
        if kwarguments['plot_skels']:
            skeleton_i = skeletons[i]
            ax.plot(skeleton_i.x, skeleton_i.z, 'o', color='k', markersize=1, alpha=0.1)

        if kwarguments['plot_synapses']:
            synapses_i = synapses[i]
            synapses_pre_i = synapses_i[(synapses_i.type=='pre') & (synapses_i.confidence>min_conf_pre)]
            synapses_post_i = synapses_i[(synapses_i.type=='post') & (synapses_i.confidence>min_conf_post)]
#                 ax.scatter(synapses_i.x, synapses_i.z, s=1, c='c', alpha=0.5)
            ax.scatter(synapses_pre_i.x, synapses_pre_i.z, s=2, c='r', alpha=0.5)
            ax.scatter(synapses_post_i.x, synapses_post_i.z, s=2, c='c', alpha=0.5)

        if kwarguments['plot_hull_synapses']:
            synapses_i = synapses[i]
            hulls_synapses_i = hulls_synapses[i]
            ax.plot(synapses_i.x.values[hulls_synapses_i.vertices], synapses_i.z.values[hulls_synapses_i.vertices], '-', color=rand_cols[i], linewidth=3)
            x_first = synapses_i.x.values[hulls_synapses_i.vertices[0]]
            x_last = synapses_i.x.values[hulls_synapses_i.vertices[-1]]
            z_first = synapses_i.z.values[hulls_synapses_i.vertices[0]]
            z_last = synapses_i.z.values[hulls_synapses_i.vertices[-1]]
            ax.plot([x_last, x_first], [z_last, z_first], '-', color=rand_cols[i], linewidth=3)

        ax.set_title(str(bodyIds[i]))
        ax.set_aspect('equal')
        ax.axis([0, 6000, 20000, 37000])

        if kwarguments['axis_on'] == False:
            ax.axis('off')


def get_bin_edges(rnge,binsize):
    num_bins = int(abs(rnge[1]-rnge[0])/binsize)
    binedges = np.linspace(rnge[0],rnge[1]-binsize,num_bins, dtype='int')
    return binedges, num_bins


def compute_2d_histogram(x, y, xrnge, yrnge, binsize):
    xbinedges, xnum_bins = get_bin_edges(xrnge,binsize)
    ybinedges, ynum_bins = get_bin_edges(yrnge,binsize)

    hist_2D = np.zeros((ynum_bins, xnum_bins))
     
    for i, bin_id in enumerate(ybinedges):
        x_i = x[(y >= bin_id) & (y < bin_id+binsize)]

        hist, bin_edges = np.histogram(x_i, range=xrnge, bins=xnum_bins)

    #         print(hist)
        hist_2D[i,:] = np.copy(hist)

    hist_max = np.max(hist_2D)
    hist_2D_norm = hist_2D/hist_max

    return hist_2D_norm, hist_2D


def check_pts_in_roi(pts_list, roi):
    output_bool = True
    for i, pt in enumerate(pts_list):
        # x coordinate
        if pt[0] < roi[0][0] or pt[0] > roi[0][1]:
            output_bool = False
        # y coordinate
        if pt[1] < roi[1][0] or pt[1] > roi[1][1]:
            output_bool = False
        # z coordinate
        if pt[2] < roi[2][0] or pt[2] > roi[2][1]:
            output_bool = False
    return output_bool


def check_pts_in_roi_2(pts_list, roi):
    output_bools = []
    for i, pt in enumerate(pts_list):
        output_bool = True
        # x coordinate
        if pt[0] < roi[0][0] or pt[0] > roi[0][1]:
            output_bool = False
        # y coordinate
        if pt[1] < roi[1][0] or pt[1] > roi[1][1]:
            output_bool = False
        # z coordinate
        if pt[2] < roi[2][0] or pt[2] > roi[2][1]:
            output_bool = False
        output_bools.append( output_bool )
    return output_bools


def return_point(df, rowId_i):
    x = df[df.rowId==rowId_i].x.values[0]
    y = df[df.rowId==rowId_i].y.values[0]
    z = df[df.rowId==rowId_i].z.values[0]
    
    point = np.array([x, y, z])
    radius = df[df.rowId==rowId_i].radius.values[0]
    return point, radius


def plot_skeleton_i(ax, df, rowId_i, link_i, plot_surface_bool, plot_skeleton_bool, roi, col_s):
    #axis and radius
    p0, R = return_point(df,rowId_i)
    p1, R1 = return_point(df,link_i)

    if check_pts_in_roi([p0, p1], roi):
        if plot_surface_bool:
    #         R = 5
            #vector in direction of axis
            v = p1 - p0
            #find magnitude of vector
            mag = norm(v)
            #unit vector in direction of axis
            v = v / mag
            #make some vector not in the same direction as v
            not_v = np.array([1, 0, 0])
            if (v == not_v).all():
                not_v = np.array([0, 1, 0])
            #make vector perpendicular to v
            n1 = np.cross(v, not_v)
            #normalize n1
            n1 /= norm(n1)
            #make unit vector perpendicular to v and n1
            n2 = np.cross(v, n1)
            #surface ranges over t from 0 to length of axis and 0 to 2*pi
            t = np.linspace(0, mag, 2)
            theta = np.linspace(0, 2 * np.pi, 8)
            #use meshgrid to make 2d arrays
            t, theta = np.meshgrid(t, theta)
            #generate coordinates for surface
            X, Y, Z = [p0[i] + v[i] * t + R * np.sin(theta) * n1[i] + R * np.cos(theta) * n2[i] for i in [0, 1, 2]]
            ax.plot_surface(X, Y, Z, color=col_s, alpha=1.0)

        if plot_skeleton_bool:
            ax.plot(*zip(p0, p1), color = col_s)
            
        
def plot_synapses_hist2d(df_synapses, description, **kwargs):
    kwarguments = {'xlims': (0, 16000),
               'ylims': (0, 16000),
               'zlims': (20000, 36000),
               'azim': 140,
               'dsize': 10,
               'dsize3d': 0.5,
               'alpha2D': 0.5,
               'alpha_addsynapses': 0.05,
               'alpha3D': 0.05,
               'zrange': (0, 500000),
               'addskel': None,
               'addsynapses': None,
               'binsize': 50}
    kwarguments.update(kwargs)

    dsize=kwarguments['dsize']
    dsize3d=kwarguments['dsize3d']
    alpha2D = kwarguments['alpha2D']
    alpha3D = kwarguments['alpha3D']
    alpha_addsynapses = kwarguments['alpha_addsynapses']
    zrange = kwarguments['zrange']
    xlims = kwarguments['xlims']
    ylims = kwarguments['ylims']
    zlims = kwarguments['zlims']
    binsize = kwarguments['binsize']

    cmap_pre = ListedColormap(np.array( sns.dark_palette("red") ))
    cmap_post = ListedColormap(np.array( sns.dark_palette("green") )    )

    if kwarguments['addsynapses'] is not None:
        addsynapses = kwarguments['addsynapses']
        addsynapses = addsynapses[(addsynapses.z>zrange[0]) & (addsynapses.z<zrange[1])]
        addsynapses_pre = addsynapses[addsynapses.type=='pre']
    
    xlims2d = (0, 4000)
    ylims2d = (8000, 14000)
    zlims2d = zrange

    df_synapses_pre = df_synapses[df_synapses.type=='pre']
    df_synapses_post = df_synapses[df_synapses.type=='post']

    df_synapses_pre_zrange = df_synapses_pre[(df_synapses_pre.z>zrange[0]) & (df_synapses_pre.z<zrange[1])]
    df_synapses_post_zrange = df_synapses_post[(df_synapses_post.z>zrange[0]) & (df_synapses_post.z<zrange[1])]


    fig = plt.figure(figsize=(15,6))
    
    # 3D plot
    ax_3d = plt.subplot(1,2,1, projection='3d', proj_type = 'ortho')
    ax = ax_3d
    
    ax.add_collection3d( ax.fill_between((xlims[0],xlims[1]),zrange[0],zrange[1], color='r', alpha=0.05),ylims[0], zdir='y' )
    ax.add_collection3d( ax.fill_between((ylims[0],ylims[1]),zrange[0],zrange[1], color='r', alpha=0.05),xlims[1], zdir='x' )

    if kwarguments['addskel'] is not None:
        roi = [xlims, ylims, zlims]
        plot_surface_bool = True
        plot_skeleton_bool = False
        skel_pd = kwarguments['addskel']
        for i, rowId_i in enumerate(skel_pd.rowId):
            try:
                progress(i, skel_pd.rowId.size, 'plotting skeleton (takes long)')
            except:
                print('...plotting skeleton')
                
            link_i = int(skel_pd[skel_pd.rowId==rowId_i].link)
            if link_i > 0:
                plot_skeleton_i(ax, skel_pd, rowId_i, link_i, plot_surface_bool, plot_skeleton_bool, roi, 'm')

    print('')
    print('...plotting 3D data')
    
    ax.scatter(df_synapses_pre.x, df_synapses_pre.y, df_synapses_pre.z*0+zlims[1], color='k', s=dsize3d, alpha=alpha3D)

    ax.scatter(df_synapses_pre.x, df_synapses_pre.y, df_synapses_pre.z, color='r', s=dsize3d, alpha=alpha3D, label='pre')
    ax.scatter(df_synapses_post.x, df_synapses_post.y, df_synapses_post.z, color='g', s=dsize3d, alpha=alpha3D, label='post')

    ax.scatter(df_synapses_pre.x, df_synapses_pre.y, df_synapses_pre.z*0+zlims[1], color='k', s=dsize3d, alpha=alpha3D)
    ax.scatter(df_synapses_post.x, df_synapses_post.y, df_synapses_post.z*0+zlims[1], color='k', s=dsize3d, alpha=alpha3D)
    ax.scatter(df_synapses_pre.x, df_synapses_pre.y*0+ylims[0], df_synapses_pre.z, color='k', s=dsize3d, alpha=alpha3D)
    ax.scatter(df_synapses_post.x, df_synapses_post.y*0+xlims[0], df_synapses_post.z, color='k', s=dsize3d, alpha=alpha3D)
    ax.scatter(df_synapses_pre.x*0+xlims[1], df_synapses_pre.y, df_synapses_pre.z, color='k', s=dsize3d, alpha=alpha3D)
    ax.scatter(df_synapses_post.x*0+xlims[1], df_synapses_post.y, df_synapses_post.z, color='k', s=dsize3d, alpha=alpha3D)
    ax.legend(markerscale=5.0)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_zlim(zlims)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(elev=30, azim=kwarguments['azim']) #
    ax.invert_zaxis()


    # 2D plot
    axes_2d = plt.subplot(1,2,2)
    ax = axes_2d
    print('')
    print('...plotting 2D data')
    
    im_shape = (get_bin_edges(ylims2d,binsize)[1], get_bin_edges(xlims2d,binsize)[1])
    image = np.zeros((*im_shape,3))
    if kwarguments['addsynapses'] is not None:
        hist_2d_all_synapses, _ = compute_2d_histogram(x=addsynapses_pre.x.values, y=addsynapses_pre.y.values, xrnge=xlims2d, yrnge=ylims2d, binsize=binsize)
        for i in range(3):
            image[:,:,i] = hist_2d_all_synapses
    hist_2d_pre_norm, hist_2d_pre = compute_2d_histogram(x=df_synapses_pre_zrange.x.values, y=df_synapses_pre_zrange.y.values, xrnge=xlims2d, yrnge=ylims2d, binsize=binsize)
    hist_2d_post_norm, hist_2d_post = compute_2d_histogram(x=df_synapses_post_zrange.x.values, y=df_synapses_post_zrange.y.values, xrnge=xlims2d, yrnge=ylims2d, binsize=binsize)
    image[:,:,0] = hist_2d_pre_norm
    image[:,:,1] = hist_2d_post_norm
    im_pre = ax.imshow(hist_2d_pre, cmap=cmap_pre)
    im_post = ax.imshow(hist_2d_post, cmap=cmap_post)
    ax.imshow(image,origin='lower')
    if np.sum(hist_2d_pre) > 1.0:
        cbar = ax.figure.colorbar(im_pre, ax=ax, shrink=0.4)
    if np.sum(hist_2d_post) > 1.0:
        cbar = ax.figure.colorbar(im_post, ax=ax, shrink=0.4)

    text_posx = im_shape[1]-im_shape[1]/20.0
    text_posy_0 = im_shape[0]-im_shape[0]/30.0
    text_posy_1 = im_shape[0]-2*im_shape[0]/30.0
    text_posy_2 = im_shape[0]-3*im_shape[0]/30.0
    ax.text(text_posx,text_posy_2,'all LOP synapses', color='b', fontsize=10)
    ax.text(text_posx,text_posy_1,'presynaptic', color='r', fontsize=10)
    ax.text(text_posx,text_posy_0,'postsynaptic', color='g', fontsize=10)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.invert_yaxis()
    ax.invert_xaxis()

    fig.suptitle(description, fontsize=18)

    return fig
    
    
def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()  # As suggested by Rom Ruben (see: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113#comment50529068_27871113)
    

def get_overlap_matrix(data_pre, description_pre, data_post, description_post, **kwargs):
    kwarguments = {'DNs_identified': None,
               'microns_per_pixel': 8/1000.0}
    kwarguments.update(kwargs)

    microns_per_pixel = kwarguments['microns_per_pixel']
    
    # replace generic neuron name by identified neuron name
    if kwarguments['DNs_identified'] is not None:
        DNs_identified = kwarguments['DNs_identified']
        for index, row in DNs_identified.iterrows():
            row_bodyId = row['bodyId']
            row_name = row['name']
#             replace_index = data_post.index(row_bodyId)
#             description_post[replace_index] = row_name
            replace_indices = getIndexPositions(data_post, row_bodyId)
            for idx in replace_indices:
                description_post[idx] = row_name

    # data: list with multiple lists each including bodyIds of a defined cell type
    overlap_matrix = np.zeros((len(data_pre), len(data_post)))

    num_connections = len(data_pre) * len(data_post)

    count = 0
    for i, data_pre_i in enumerate(data_pre):
        for j, data_post_j in enumerate(data_post):
            count = count+1

            hull_i_pts = data_pre_i.points
            hull_i_vert = data_pre_i.vertices

            hull_j_pts = data_post_j.points
            hull_j_vert = data_post_j.vertices

            P_i = Polygon(hull_i_pts[hull_i_vert])
            P_j = Polygon(hull_j_pts[hull_j_vert])

            P_intersection = P_i.intersection(P_j)
            area_px = P_intersection.area
            
            area_microns = (np.sqrt(area_px)*microns_per_pixel)**2

#             if i == j:
#                 overlap_matrix[i,j] = 0
#             else:
            overlap_matrix[i,j] = area_microns

#            print(str(round(count/num_connections*100)), ' percent complete', end='\r')
            progress(count, num_connections, 'calculate overlap')

    connectivity_dict = {}
    connectivity_dict['connectivity_matrix'] = overlap_matrix
    connectivity_dict['description_pre'] = description_pre
    connectivity_dict['description_post'] = description_post
    connectivity_dict['data_pre'] = data_pre
    connectivity_dict['data_post'] = data_post
    return connectivity_dict



def plot_skeleton(skel_pd, **kwargs):
    kwarguments = {'xlims': (0, 16000),
               'ylims': (0, 16000),
               'zlims': (20000, 36000),
               'azim': 140,
               'elev': 30,
               'dsize': 10,
               'dsize3d': 0.5,
               'alpha2D': 0.5,
               'alpha_addsynapses': 0.05,
               'alpha3D': 0.05,
               'zrange': (0, 500000),
               'addsynapses': None,
               'plot_surface_bool': False,
               'plot_skeleton_bool': True,
               'axis_onoff': 'on'}
    kwarguments.update(kwargs)
    
    plot_surface_bool = kwarguments['plot_surface_bool']
    if plot_surface_bool:
        plot_skeleton_bool = False
    else:
        plot_skeleton_bool = kwarguments['plot_skeleton_bool']
    dsize=kwarguments['dsize']
    dsize3d=kwarguments['dsize3d']
    alpha2D = kwarguments['alpha2D']
    alpha3D = kwarguments['alpha3D']
    alpha_addsynapses = kwarguments['alpha_addsynapses']
    zrange = kwarguments['zrange']
    xlims = kwarguments['xlims']
    ylims = kwarguments['ylims']
    zlims = kwarguments['zlims']

    fig = plt.figure(figsize=(20,10))

    # 3D plot
    ax_3d = plt.subplot(1,2,1, projection='3d', proj_type = 'ortho')
    ax = ax_3d

    roi = [xlims, ylims, zlims]
    
    link_count = 0
    missing_links_points = []
    end_point_rowIds = list( skel_pd[~skel_pd.rowId.isin(skel_pd.link)].rowId.values )
    end_points = []
    for i, rowId_i in enumerate(skel_pd.rowId):
        try:
            progress(i, skel_pd.rowId.size, 'plotting skeleton (takes long)')
        except:
            print('...plotting skeleton')
            
        link_i = int(skel_pd[skel_pd.rowId==rowId_i].link)
        if link_i > 0:
            plot_skeleton_i(ax, skel_pd, rowId_i, link_i, plot_surface_bool, plot_skeleton_bool, roi, 'c')
        else:
            point, radius = return_point(skel_pd, rowId_i)
            missing_links_points.append(point)
            link_count += 1
        if rowId_i in end_point_rowIds:
            point, radius = return_point(skel_pd, rowId_i)
            end_points.append(point)
            
    missing_links_points = np.array( missing_links_points )
    ax.plot(missing_links_points[:,0], missing_links_points[:,1], missing_links_points[:,2], 'o', color='red', markersize=8, alpha=0.5, label='link = -1')

    end_points_bools_in_roi = check_pts_in_roi_2(end_points, roi) # return list of booleans to indicate whether points are inside or outside roi
    end_points = np.array( end_points )[end_points_bools_in_roi] # return only endpoints points within roi
    ax.plot(end_points[:,0], end_points[:,1], end_points[:,2], 'o', color='blue', markersize=4, alpha=0.5, label='end points')
    
    print('')
    print('...plotting 3D data')

    ax.legend()
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_zlim(zlims)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(elev=kwarguments['elev'], azim=kwarguments['azim']) #
    ax.invert_zaxis()
    ax.axis(kwarguments['axis_onoff'])

#    fig.suptitle(description, fontsize=18)

    return fig
