# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 11:32:49 2022

@author: donmez
"""

#importing needed libraries

import numpy as np
import skimage.segmentation
from matplotlib import pyplot as plt


#%%
####################################################
#  READ MULTISPECTRAL SENTINEL 2 TIF IMAGE
####################################################
# from osgeo import gdal
fp = r'C:/Users/doenmez/Documents/gnn/Graph-Segmentation/nrw_may_median/muenster1024/med_may2021_muenster_1024p_2-0000000000-0000001024.tif'

# read multispectral sentinel 2 image
import rasterio
with rasterio.open(
    fp, mode="r", nodata=0) as src:
    # read in the array, band 3 first, then band 2, then band 1
    arr = src.read()
    # the array has three bands
    print("Array shape:", arr.shape)
    # look at the profile, despite setting nodata=0, there still isn't a nodata value 
    # this is because we need to update the profile and write out a new image with
    # nodata set
    profile = src.profile
    print(profile)


img_dim = arr.shape[2]

#%% 
####################################################
#  PERFORM SEGMENTATION
####################################################
arr2 = np.transpose(arr, [1,2,0]) 

ndvi = (arr2[:,:,7]-arr2[:,:,3])/(arr2[:,:,7]+arr2[:,:,3])

# #performing segmentation
res1 = skimage.segmentation.felzenszwalb(arr2, scale=82100)
print(res1.max())


#%% 
####################################################
# CALCULATE MEAN OF SPECTRAL BANDS FOR EACH SEGMENT
####################################################
arrseg = np.empty((img_dim, img_dim, 9))        # boş array oluşturduk image'a segment bilgisi eklemek için 

print(arrseg.shape)  # prints (2, 3, 4)

arrseg[:,:,0:8]=arr2                    # imageın bantları
arrseg[:,:,8]=res1                      # segment bilgisi

# take the means of spectral bands for each segment to represent the segment
meanfeat=np.empty((res1.max(),arrseg.shape[2]))    # boş array oluşturduk içine her segment için 
meanfeat[:,0]=range(res1.max())                    # her bandın ortalamasını koyucaz


          

import time
start_time = time.time()

for seg in range(res1.max()):
    for feat in range(arrseg.shape[2] - 1):
        mask = (arrseg[:, :, 8] == seg)
        selected = arrseg[mask, :]
        mean = np.mean(selected[:, feat])

        # Append the mean to the list of means
        meanfeat[seg, feat + 1] = mean

    # Print intermediate steps after every 100 segments
    if (seg + 1) % 100 == 0:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Processed {seg + 1} segments. Elapsed time: {elapsed_time} seconds")

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Total time taken: {elapsed_time} seconds")     



# Following better when  memory available
# # Calculate mean of spectral bands for each segment without explicit loop
# segment_means = np.zeros((res1.max(), arrseg.shape[2] - 1))

# for feat in range(arrseg.shape[2] - 1):
#     band_data = arrseg[:, :, feat]
#     unique_segments = np.unique(arrseg[:, :, 8])

#     # Create a boolean mask for each segment
#     masks = (arrseg[:, :, 8][:, :, None] == unique_segments)

#     # Use broadcasting to calculate mean for each segment
#     segment_means[:, feat] = np.nanmean(np.where(masks, band_data[:, :, None], np.nan), axis=(0, 1))

# # Append segment indices to the calculated means
# meanfeat = np.column_stack((unique_segments, segment_means))




########### SAVE FILES ################

# import pickle
# # Save all variables to one file
# all_variables = {
#     'arrseg': arrseg,
#     'res1': res1,
#     'meanfeat': meanfeat,
#     'file_directory': r'C:/Users/doenmez/Documents/gnn/Graph-Segmentation/',
#     # Add more variables as needed
# }

# with open('all_variables_muenster1024_fieldindex.pkl', 'wb') as file:
#     pickle.dump(all_variables, file)
    

            
#%%  
####################################################      
# EXTRACT NEIGHBOR INFORMATION FOR EACH SEGMENT
####################################################

# function to extract neighbor information
def get_neighbors(array, i, j):
    # Check if i and j are within the bounds of the array
    if i < 0 or i >= len(array) or j < 0 or j >= len(array[i]):
        return None
    neighbors = []
    # Check for neighbors to the north
    if i > 0:
        neighbors.append(array[i-1][j])
    # Check for neighbors to the south
    if i < len(array) - 1:
        neighbors.append(array[i+1][j])
    # Check for neighbors to the west
    if j > 0:
        neighbors.append(array[i][j-1])
    # Check for neighbors to the east
    if j < len(array[i]) - 1:
        neighbors.append(array[i][j+1])
    return neighbors

# use get_neighbors()function for each pixel
array=res1
nbhd=np.full((array.shape[0]*array.shape[1],5),np.nan)

for x in range(array.shape[0]):
    for y in range(array.shape[1]):
        neighbors = get_neighbors(array,x,y)
        nbhd[x*img_dim+y,0]=array[x,y]
        if len(neighbors)==2:
            nbhd[x*img_dim+y,1:3]=neighbors
        elif len(neighbors)==3:
            nbhd[x*img_dim+y,1:4]=neighbors
        else:
            nbhd[x*img_dim+y,1:5]=neighbors
       
        
# get unique values to obtain neighboring information of each segment
nbhd_uniq=np.unique(nbhd,axis=0)
nhbd_sum=np.full((res1.max(),33),np.nan)

for seg in range(res1.max()):
    mask=(nbhd_uniq[:,0]==seg)
    selected=nbhd_uniq[mask,:]

    unique_values, counts = np.unique(selected, return_counts=True)
    unique_values = unique_values[~np.isnan(unique_values)]
    nhbd_sum[seg,0]=seg
    indices = np.where(unique_values == seg)
    neigh = np.delete(unique_values, indices)
    for i in range(33):
        if len(neigh)==i:
            nhbd_sum[seg,1:i+1]=neigh


#%%
####################################################
#  READ INVEKOS GROUNDTRUTH TIF IMAGE
####################################################
from skimage import io

# Read the TIFF image with invekos ground truth information
invekos = io.imread('C:/Users/doenmez/Documents/gnn/Graph-Segmentation/nrw_may_median/muenster1024/muenster1024_0000000000-0000001024_invekos_rasterized_fieldindex.tif')
classes = invekos[:,:,1]
fields = invekos[:,:,0]

# Convert the image to a NumPy array
segclassfield = np.empty((img_dim, img_dim, 3))


#%%
####################################################
#  ASSIGN CROP CLASS TO EACH SEGMENT BY FINDING WHICH CLASS OCCUPIES MAJORITY OF PIXELS IN THE SEGMENT
####################################################
segclassfield[:,:,0]=res1
segclassfield[:,:,1]=classes
segclassfield[:,:,2]=fields

classno_82100 = np.empty((res1.max(),3))
import scipy.stats
for seg in range(res1.max()):
    mask=(segclassfield[:,:,0]==seg)
    selected=segclassfield[mask,:]
    selected_major=scipy.stats.mode(selected[:,1])[0]
    # selected_unique=np.unique(selected,axis=0)
    classno_82100[seg,0]=seg
    classno_82100[seg,1]=selected_major
    
for seg in range(res1.max()):
    mask=(segclassfield[:,:,0]==seg)
    selected=segclassfield[mask,:]
    selected_major=scipy.stats.mode(selected[:,2])[0]
    # selected_unique=np.unique(selected,axis=0)
    classno_82100[seg,0]=seg
    classno_82100[seg,2]=selected_major

#%%
####################################################
#  TRANFORM THE DATA INTO A GRAPH
####################################################
import networkx as nx

# Create an empty graph
G = nx.Graph()

# Add the nodes to the graph
for i in range(res1.max()):
    segment = nhbd_sum[i, 0]
    G.add_node(segment)

# Add the edges to the graph
for i in range(res1.max()-1):
    segment = nhbd_sum[i, 0]
    nbhdnum0 = nhbd_sum[i, 1:]
    nbhdnum = nbhdnum0[~np.isnan(nbhdnum0)]
    
    for j in range(nbhdnum.shape[0]-1):
        neighbor = nbhdnum[j]
        G.add_edge(segment, neighbor)


# Print the number of nodes and edges
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")

allfeatures = np.column_stack((meanfeat,  classno_82100[:,1], classno_82100[:,2]))

# Define the names of the features
feature_names = ["B2", "B3", "B4","B5", "B6", "B7","B8", "B8A","clss","field" ]

# Iterate over the nodes in the graph
for i, node in enumerate(G.nodes):
  # Get the features for this node from the NumPy array
  node_features = dict(zip(feature_names, allfeatures[i,1:11]))
  
  # Add the features to the node
  nx.set_node_attributes(G, {node: node_features})

# Print the nodes and their features
# print(G.nodes(data=True))


for node, attributes in G.nodes(data=True):
  # Print the number of features for each node
  print(f"Node {node} has {len(attributes)} features")


# nx.draw(G)
def print_graph_info(graph):
  print("Directed graph:", graph.is_directed())
  print("Number of nodes:", graph.number_of_nodes())
  print("Number of edges:", graph.number_of_edges())
# print_graph_info(G)


#%% ####################################################################################################################
# SPLIT THE DATA TO TRAINING AND TEST SET IN A WAY THAT NO INVEKOS FIELD HAVE SEGMENTS IN BOTH TRAINING AND TEST SET
########################################################################################################################
import torch
import torch.nn as nn
import torch.optim as optim
# from torch_sparse import SparseTensor
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

# Tranform the data from Netwokx graph to torch tensor with edge indices and node features
edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()

# Convert edge_index to PyTorch Geometric's format
edge_index = from_networkx(G).edge_index
num_classes = len(list(set(allfeatures[:, 9])))
x = np.unique(allfeatures[:, 9])

# change the class numbers from invekos standarts to 0:no of class
dict_map = {n:i for (i, n) in enumerate(x)}
temp_cls = [dict_map[x_] for x_ in allfeatures[:, 9]]
allfeatures[:, 9] = temp_cls 



# Step 2: Define Sparse Node Features
node_features = torch.tensor(allfeatures[:, 1:9], dtype=torch.float)
class_labels = torch.tensor(allfeatures[:, 9], dtype=torch.long)


# remove 0 from field indices. 0 means no ground truth data
field_indices = allfeatures[:, 10]
field_indices = field_indices[field_indices!=0]
field_indices = list(dict.fromkeys(field_indices.tolist()))

num_fields=len(list(set(field_indices)))
# Define the ratio of samples to be used for training (e.g., 80%)
train_ratio = 0.8
train_samples = int(train_ratio * num_fields)



# Convert it to a PyTorch tensor
allfeatures_tensor = torch.tensor(allfeatures, dtype=torch.float)


# Randomly shuffle the indices
# Set a random seed for reproducibility
import random
random.seed(2)
random.shuffle(field_indices)

# Define the ratio of samples to be used for training (e.g., 80%)
train_samples = int(train_ratio * len(field_indices))

# Split the indices into train and test
train_field_indices = field_indices[:train_samples]
test_field_indices = field_indices[train_samples:]

# Use the shuffled indices to split allfeatures into train and test sets
train_set = allfeatures_tensor[[i for i in range(len(allfeatures)) if allfeatures[i, 10] in train_field_indices]]
test_set = allfeatures_tensor[[i for i in range(len(allfeatures)) if allfeatures[i, 10] in test_field_indices]]

# Extract features and labels from the training and test sets
train_features = train_set[:, 1:10]
train_labels = train_set[:, 9]

test_features = test_set[:, 1:10]
test_labels = test_set[:, 9]

# Print the shapes of train and test sets
print("Train set shape:", train_features.shape, train_labels.shape)
print("Test set shape:", test_features.shape, test_labels.shape)


# numpy falan yapıp train setin segment indexlerini al ordan train_data oluştur. test set için aynısı 
clss0_indices = allfeatures[:, 0][np.isin(allfeatures[:, 9], [0, 1, 2,3,4, 5, 7,9, 11,12, 15, 16, 19, 20,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39])]
clss0_indices= torch.tensor(clss0_indices)

train_indic_unfltrd = train_set[:,0]

# remove those from train and test indices
train_indices= train_indic_unfltrd[~train_indic_unfltrd.unsqueeze(1).eq(clss0_indices).any(1)]



test_indic_unfltrd = test_set[:,0]
test_indices= test_indic_unfltrd[~test_indic_unfltrd.unsqueeze(1).eq(clss0_indices).any(1)]


train_indices= train_indices.tolist()
test_indices = test_indices.tolist()


# Create train and test data tensors
gnn_data = Data(x=node_features, edge_index=edge_index, y=class_labels)
train_data = gnn_data

train_data.y = train_data.y[train_indices]

import matplotlib.pyplot as pltt
pltt.figure()
pltt.hist(train_data.y, np.arange(1, 40))
pltt.xticks(np.arange(1, 40))

# Filter the training data to exclude class=0 segments
# train_data.y=train_data.y[train_data.y != 0]



gnn_data = Data(x=node_features, edge_index=edge_index, y=class_labels)
test_data = gnn_data
test_data.y = test_data.y[test_indices]


#%% ####################################################################################################################
# Perform GNN classification
########################################################################################################################

# Define the GNN Model with Sparse Operations
class SparseGNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SparseGNNModel, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.tanh(x)
        x = self.conv2(x, edge_index)
        return x

num_classes = len(list(set(allfeatures[:, 9])))

# Initialize the GNN model

model = SparseGNNModel(input_dim=8, hidden_dim=512, output_dim=num_classes)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# import torch.nn.functional as F 
data=train_data

# Define training loop
def train_sparse(model, data, optimizer, criterion, train_indices, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(data)
        train_labels = data.y
        loss = criterion(output[train_indices], train_labels)
        loss.backward()
        optimizer.step()

# Train the model with the train_data
train_sparse(model, train_data, optimizer, criterion, train_indices, num_epochs=400)


# Evaluate the Model with train set
model.eval()
with torch.no_grad():
    output_tr = model(train_data)
    train_labels = train_data.y

    # Calculating accuracy:
    predictions_tr = output_tr[train_indices].argmax(dim=1)
    accuracy_tr = (predictions_tr == train_labels).sum().item() / len(train_labels)
    print(f"Train Accuracy: {accuracy_tr * 100:.2f}%")



# Evaluate the Model with test set
model.eval()
with torch.no_grad():
    output = model(test_data)
    test_labels = test_data.y
    # You can use your preferred evaluation metrics to assess the model's performance.
    # For example, calculating accuracy:
    predictions = output[test_indices].argmax(dim=1)
    accuracy = (predictions == test_labels).sum().item() / len(test_labels)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")





# Hyperparameter Tuning
import time

# Loop over hyperparameters
for num_epochs in [250, 300, 350, 400, 500, 1000]:
    for lr in [ 0.01, 0.05, 0.001, 0.005, 0.0001]:
        for hidden_dim in [256, 512, 1024]:
            start_time = time.time()

            # Create the model
            model = SparseGNNModel(input_dim=8, hidden_dim=hidden_dim, output_dim=num_classes)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            # Train the model
            train_sparse(model, train_data, optimizer, criterion, train_indices, num_epochs=num_epochs)

            # Evaluate on the training set
            model.eval()
            with torch.no_grad():
                output_tr = model(train_data)
                train_labels = train_data.y
                predictions_tr = output_tr[train_indices].argmax(dim=1)
                accuracy_tr = (predictions_tr == train_labels).sum().item() / len(train_labels)

            # Evaluate on the test set
            model.eval()
            with torch.no_grad():
                output = model(test_data)
                test_labels = test_data.y
                predictions = output[test_indices].argmax(dim=1)
                accuracy = (predictions == test_labels).sum().item() / len(test_labels)

            end_time = time.time()
            elapsed_time = end_time - start_time

            # Print the results
            print(f"Epochs: {num_epochs}, LR: {lr}, Hidden Dim: {hidden_dim}")
            print(f"Train Accuracy: {accuracy_tr * 100:.2f}%")
            print(f"Test Accuracy: {accuracy * 100:.2f}%")
            print(f"Time Elapsed: {elapsed_time:.2f} seconds")
            print("=" * 50)


