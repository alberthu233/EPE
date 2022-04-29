import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import uproot as ur
import awkward as ak
import time as t
import copy
import scipy.constants as spc
import os
print("Awkward version: "+str(ak.__version__))
print("Uproot version: "+str(ur.__version__))


# data and branches 
track_branches = ['trackEta_EMB1', 'trackPhi_EMB1', 'trackEta_EMB2', 'trackPhi_EMB2', 'trackEta_EMB3', 'trackPhi_EMB3',
                  'trackEta_TileBar0', 'trackPhi_TileBar0', 'trackEta_TileBar1', 'trackPhi_TileBar1',
                  'trackEta_TileBar2', 'trackPhi_TileBar2']

event_branches = ["cluster_nCells", "cluster_cell_ID", "cluster_cell_E", 'cluster_nCells', "nCluster", "eventNumber",
                  "nTrack", "nTruthPart", "truthPartPdgId", "cluster_Eta", "cluster_Phi", 'trackPt', 'trackP',
                  'trackMass', 'trackEta', 'trackPhi', 'truthPartE', 'cluster_ENG_CALIB_TOT', "cluster_E", 'truthPartPt']

geo_branches = ["cell_geo_ID", "cell_geo_eta", "cell_geo_phi", "cell_geo_rPerp", "cell_geo_sampling"]

# make calorimeter layer dictionary
sampling_layer_track_list = [1,1,2,2,3,3,12,12,13,13,14,14]
track_layer_dict = dict(zip(track_branches, sampling_layer_track_list))

# define all function used in the code
def DeltaR(coords, ref):
    ''' Straight forward function, expects Nx2 inputs for coords, 1x2 input for ref '''
    ref = np.tile(ref, (len(coords[:,0]), 1))
    DeltaCoords = np.subtract(coords, ref)
    return np.sqrt(DeltaCoords[:,0]**2 + DeltaCoords[:,1]**2) 

def track_av(_arr):
    ''' Expects a (6,2) np array for the barrel layers in order of eta, phi '''
    _av_Eta = np.sum(_arr[:,0])/6
    _av_Phi = np.sum(_arr[:,1])/6
    return np.array([_av_Eta, _av_Phi])

def find_max_dim(_events, _event_dict):
    ''' This function is designed to return the sizes of a numpy array such that we are efficient
    with zero padding. Please feel free to write this faster, it can be done. Notes: we add six
    to the maximum cluster number such that we have room for track info.
    Inputs:
        _events: filtered list of events to choose from in an Nx3 format for event, track, cluster index 
        _event_tree: the event tree dictionary
    Returns:
        3-tuple consisting of (number of events, maximum cluster_size, 5), 5 because of how we have structured
        the X data format in energyFlow to be Eta, Phi, Energy, sampling layer, track flag,
        _cluster_ENG_CALIB_TOT, turthPartE '''
    _nEvents = len(_events)
    _max_clust = 0
    for i in range(_nEvents):
        _evt = _events[i,0]
        _clust_idx = _events[i,2]
        _num_in_clust = _event_dict['cluster_nCells'][_evt][_clust_idx]
        if _num_in_clust > _max_clust:
            _max_clust = _num_in_clust

    return (_nEvents, _max_clust+6, 6)

def find_max_dim_tuple(events, event_dict):
    nEvents = len(events)
    max_clust = 0
    
    for i in range(nEvents):
        event = events[i,0]
        track_nums = events[i,1]
        clust_nums = events[i,2]
        
        clust_num_total = 0
        # set this to six for now to handle single track events, change later
        track_num_total = 6
        
        # Check if there are clusters, None type object may be associated with it
        if clust_nums is not None:
            # Search through cluster indices
            for clst_idx in clust_nums:
                nInClust = len(event_dict['cluster_cell_ID'][event][clst_idx])
                # add the number in each cluster to the total
                clust_num_total += nInClust
                
#         if track_nums is not None:
#             for trck_idx in track_nums:
#                 # each track takes up six coordinates
#                 track_num_total += 6

        total_size = clust_num_total + track_num_total
        if total_size > max_clust:
            max_clust = total_size
    
    # 6 for energy, eta, phi, rperp, track flag, sample layer
    return (nEvents, max_clust, 6)
            
            

def track_coords(_idx, _event_dict, _track_dict, _track_branches):
    ''' Returns a list of numpy arrays which contain the track information in the order of
    EMB1->TileBar2. '''    
    _num_tracks = _event_dict['nTrack'][_idx]
    _track_arr = np.empty((_num_tracks,6,2))
    for i in range(_num_tracks):    
        _coords = np.empty((12,))
        j = 0
        for _key in _track_branches:
            _coords[j] = _track_dict[_key][_idx][i]
            j += 1
    _track_arr[i] = _coords.reshape(6,2)
    return _track_arr

def dict_from_event_tree(_event_tree, _branches):
    ''' The purpose for this separate function is to load np arrays where possible. '''
    _special_keys = ["nCluster", "eventNumber", "nTrack", "nTruthPart"]
    _dict = dict()
    for _key in _branches:
        if _key in _special_keys:
            _branch = _event_tree.arrays(filter_name=_key)[_key].to_numpy()
        else:
            _branch = _event_tree.arrays(filter_name=_key)[_key]
        _dict[_key] = _branch
    return _dict

def dict_from_tree_branches(_tree, _branches):
    ''' Helper function to put event data in branches to make things easier to pass to functions,
    pretty self explanatory. '''
    _dict = dict()
    for _key in _branches:
        _branch = _tree.arrays(filter_name=_key)[_key]
        _dict[_key] = _branch
    return _dict

def dict_from_tree_branches_np(_tree, _branches):
    ''' Helper function to put event data in branches to make things easier to pass to functions,
    pretty self explanatory. This always returns np arrays in the dict. '''
    _dict = dict()
    for _key in _branches:
        _branch = np.ndarray.flatten(_tree.arrays(filter_name=_key)[_key].to_numpy())
        _dict[_key] = _branch
    return _dict

def find_central_clusters(_event_dict):
    ''' Inputs:
        dictionary with events to pull cluster centers from
    Returns:
        an array of event indices with clusters with eta < .7 '''
    # list of event indices with one or more clusters in EMB1-3 or TileBar0-2
    _central_events = []
    _ak_cluster_cell_ID = _event_dict['cluster_cell_ID']
    
    for _evt_idx in range(len(_ak_cluster_cell_ID)):
        _cluster_Eta = event_dict['cluster_Eta'][_evt_idx].to_numpy()
        _cluster_Eta_mask = np.abs(_cluster_Eta) < .7
        
        if np.any(_cluster_Eta_mask):
            _central_events.append(_evt_idx)

    return np.array(_central_events)

def find_index_1D(values, dictionary):
    ''' Use a for loop and a dictionary. values are the IDs to search for. dict must be in format 
    (cell IDs: index) '''
    idx_vec = np.zeros(len(values), dtype=np.int32)
    for i in range(len(values)):
        idx_vec[i] = dictionary[values[i]]
    return idx_vec


## load data

# difine path
path_prefix = 'D:/Work/EPE/ML4pi/'
plotpath = path_prefix+'plots/'
modelpath_c = path_prefix+''
modelpath = path_prefix+''
ext_path = "H:/EPE_file_storage/"
ext_modelpath = ext_path + "Model/"
ext_datapath = 'H:/EPE_file_storage/data_storage/pipm/root/'
ext_plotpath = ext_path + "plots/"

# set up file names
print('Setting up file names...')
Nfile = 150
print('Number of files:', Nfile)
fileNames = []
file_prefix = 'user.angerami.24559744.OutputStream._000'
for i in range(1,Nfile+1):
    endstring = f'{i:03}'
    fileNames.append(file_prefix + endstring + '.root')

# setup geo dictionary
print("Loading geometry...")
geo_file = ur.open(ext_datapath+'user.angerami.24559744.OutputStream._000001.root')
CellGeo_tree = geo_file["CellGeo"]
geo_dict = dict_from_tree_branches_np(CellGeo_tree, geo_branches)


k = 1 # tally used to keep track of file number
tot_nEvts = 0 # used for keeping track of total number of events
max_nPoints = 0 # used for keeping track of the largest 'point cloud'
t_tot = 0 # total time

for file_name in fileNames:
    if not os.path.isfile(ext_datapath+file_name):
        print()
        print('File '+ext_datapath+file_name+' not found..')
        print()
        k += 1
        continue
    
    else:
        print()
        print('Working on File: '+str(file_name)+' - '+str(k)+'/'+str(Nfile))
        k += 1

    t0 = t.time()
    # find event with central clusters
    print('Finding central clusters...')
    event = ur.open(ext_datapath+file_name)
    event_tree = event["EventTree"]
    event_dict = dict_from_event_tree(event_tree, event_branches)
    central_clusters = find_central_clusters(event_dict)
    nCentralClusterEvts = len(central_clusters)
    print('Number of central clusters:', nCentralClusterEvts)

    max_cluster_size = 0
    single_cluster_only = []

    # loop over central clusters
    for i in range(nCentralClusterEvts):
        _evt = central_clusters[i]
        
        # pull cluster center
        _cluster_Eta = event_dict['cluster_Eta'][_evt].to_numpy()
        
        # check if clusters have eta < .7
        _clst_cntr_mask = np.abs(_cluster_Eta) < .7
        
        # check which cluster is the max energy, make mask
        _clstr_E = event_dict['cluster_E'][_evt].to_numpy()
        _clstr_E_mask = _clstr_E == np.max(_clstr_E)
        
        # both max energy and central
        _evt_mask = np.logical_and(_clst_cntr_mask, _clstr_E_mask)
        
        # if both are true pull the index
        if np.any(_evt_mask):
            _clstr_idx = np.argmax(_evt_mask)
            single_cluster_only.append((_evt, _clstr_idx))
            
        _current_clust_num = len(event_dict['cluster_cell_ID'][_evt][_clstr_idx])
        if _current_clust_num > max_cluster_size:
            max_cluster_size = _current_clust_num
    single_cluster_only = np.array(single_cluster_only)

    # create array
    print('Creating array...')
    Y_sc = np.empty(single_cluster_only.shape[0])
    X_sc = np.zeros((single_cluster_only.shape[0], max_cluster_size, 5))
    for i in range(len(single_cluster_only)):
        _evt = single_cluster_only[i,0]
        _clstr = single_cluster_only[i,1]
        
        _cluster_cell_ID = event_dict['cluster_cell_ID'][_evt][_clstr].to_numpy()
        _cell_geo_ID = geo_dict['cell_geo_ID']
        _cluster_Eta = event_dict['cluster_Eta'][_evt][_clstr]
        _cluster_Phi = event_dict['cluster_Phi'][_evt][_clstr]
        _nInClust = len(_cluster_cell_ID)
    #     _geo_idx = find_index_1D(_cluster_cell_ID, geo_dict['cell_geo_ID'])
        
        _cell_idx = np.zeros(_nInClust, dtype=np.int32)
        for j in range(_nInClust):
            _cell_ID = _cluster_cell_ID[j]
            _cell_idx[j] = np.where(_cell_ID == _cell_geo_ID)[0][0]
        
        X_sc[i,0:_nInClust,0] = event_dict['cluster_cell_E'][_evt][_clstr].to_numpy()
        X_sc[i,0:_nInClust,1] = geo_dict['cell_geo_eta'][_cell_idx] - _cluster_Eta
        X_sc[i,0:_nInClust,2] = geo_dict['cell_geo_phi'][_cell_idx] - _cluster_Phi
        X_sc[i,0:_nInClust,3] = geo_dict['cell_geo_rPerp'][_cell_idx]
        X_sc[i,0:_nInClust,4] = geo_dict['cell_geo_sampling'][_cell_idx]
        
        # save target
        Y_sc[i] = event_dict['cluster_ENG_CALIB_TOT'][_evt][_clstr]
    t1 = t.time()
    single_cluster_only = np.array(single_cluster_only)
    print('Max cluster size: '+str(max_cluster_size))
    print('Shape of single cluster index array: '+str(single_cluster_only.shape))
    print('Time in hours: '+str(round( (t1-t0)/3600, 2)))
    print('Time in minutes: '+str(round( (t1-t0)/60, 2)))
    print('Time in seconds: '+str(round( (t1-t0), 2)))

    # save data
    print('Saving data...')
    np.savez(ext_datapath+'clusters/single_cluster_only'+str(k-1) + '.npz', X_sc, Y_sc)