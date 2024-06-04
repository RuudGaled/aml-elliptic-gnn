import pandas as pd
import torch
import os.path as osp
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit

def load_data(data_path, noAgg=False):

    # # Read edges, features and classes from csv files
    # df_edges = pd.read_csv(osp.join(data_path, "elliptic_txs_edgelist.csv"))
    # df_features = pd.read_csv(osp.join(data_path, "elliptic_txs_features.csv"), header=None)
    # df_classes = pd.read_csv(osp.join(data_path, "elliptic_txs_classes.csv"))

    # # Name colums basing on index
    # colNames1 = {'0': 'txId', 1: "Time step"}
    # colNames2 = {str(ii+2): "Local_feature_" + str(ii+1) for ii in range(94)}
    # colNames3 = {str(ii+96): "Aggregate_feature_" + str(ii+1) for ii in range(72)}

    # colNames = dict(colNames1, **colNames2, **colNames3)
    # colNames = {int(jj): item_kk for jj, item_kk in colNames.items()}

    # # Rename feature columns
    # df_features = df_features.rename(columns=colNames)
    # if noAgg:
    #     df_features = df_features.drop(df_features.iloc[:, 96:], axis = 1)

    # # Map unknown class to '3'
    # df_classes.loc[df_classes['class'] == 'unknown', 'class'] = '3'

    # # Map unknown class to '3'
    # df_classes.loc[df_classes['class'] == 'unknown', 'class'] = '3'

    # # Merge classes and features in one Dataframe
    # df_class_feature = pd.merge(df_classes, df_features)

    # # Exclude records with unknown class transaction
    # df_class_feature = df_class_feature[df_class_feature["class"] != '3']

    # # Build Dataframe with head and tail of transactions (edges)
    # known_txs = df_class_feature["txId"].values
    # df_edges = df_edges[(df_edges["txId1"].isin(known_txs)) & (df_edges["txId2"].isin(known_txs))]

    # # Build indices for features and edge types
    # features_idx = {name: idx for idx, name in enumerate(sorted(df_class_feature["txId"].unique()))}
    # class_idx = {name: idx for idx, name in enumerate(sorted(df_class_feature["class"].unique()))}

    # # Apply index encoding to features
    # df_class_feature["txId"] = df_class_feature["txId"].apply(lambda name: features_idx[name])
    # df_class_feature["class"] = df_class_feature["class"].apply(lambda name: class_idx[name])

    # # Apply index encoding to edges
    # df_edges["txId1"] = df_edges["txId1"].apply(lambda name: features_idx[name])
    # df_edges["txId2"] = df_edges["txId2"].apply(lambda name: features_idx[name])

    #--------------------------------------------------------------------------------------------#

    # df_edges = pd.read_csv(osp.join(data_path, "edges.csv"), usecols=['clId1', 'clId2'])
    df_edges = pd.read_csv(osp.join(data_path, "edges.csv"))

    connected_components = pd.read_csv(osp.join(data_path, "connected_components.csv"))
    nodes = pd.read_csv(osp.join(data_path, "nodes.csv"), nrows=123000)
    background_nodes = pd.read_csv(osp.join(data_path, "background_nodes.csv"), nrows=123000)

    # print(sum(1 for row in csv.reader(open("background_nodes.csv"))))

    # Map ccLabel in numeric value
    connected_components.loc[connected_components['ccLabel'] == 'licit', 'ccLabel'] = '0'
    connected_components.loc[connected_components['ccLabel'] == 'suspicious', 'ccLabel'] = '1'

    print(connected_components)

    # Merge part
    merge1 = pd.merge(connected_components, nodes, on='ccId')
    df_class_feature = pd.merge(merge1, background_nodes, on='clId')


    # print("edges: \n", df_edges)
    # print("altro: \n",df_class_feature)
    
    return df_class_feature, df_edges
    
    return df_class_feature, df_edges


def data_to_pyg(df_class_feature, df_edges):

    # # Define PyTorch Geometric data structure with Pandas dataframe values
    # edge_index = torch.tensor([df_edges["txId1"].values,
    #                         df_edges["txId2"].values], dtype=torch.long)
    # x = torch.tensor(df_class_feature.iloc[:, 3:].values, dtype=torch.float)
    # y = torch.tensor(df_class_feature["class"].values, dtype=torch.long)

    # data = Data(x=x, edge_index=edge_index, y=y)
    # data = RandomNodeSplit(num_val=0.15, num_test=0.2)(data)

    #-------------------------------------------------------------------------------------------#

    edge_index = torch.tensor(df_edges.values, dtype=torch.long)
    x = torch.tensor(df_class_feature.iloc[:, 3:].values, dtype=torch.float)
    y = torch.tensor(df_class_feature["ccId"].values, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    data = RandomNodeSplit(num_val=0.15, num_test=0.2)(data)

    # print("edge_index: \n")
    # print(edge_index)


    # print("x: \n")
    # print(x)

    # print("y: \n")
    # print(y)

    # print("data: \n")
    # print(data)

    return data

def reduce_features(df, corr_min=0.9):
    print("df shape original:", df.shape)
    corr = df[df.columns[97:]].corr()
    df_feat = corr.unstack().reset_index()
    #print("df:", df.head())
    df_feat.columns = ["f1", "f2", "value"]
    df_feat = df_feat[df_feat.f1 != df_feat.f2]
    df_feat = df_feat[(df_feat.value > corr_min) | (df_feat.value < -corr_min)]
    df_feat = df_feat.reset_index(drop=True)
    #print("df_feat: ", df_feat.head())
    to_remove = []
    existent = []
    for index, row in df_feat.iterrows():
      new = (row.f1, row.f2)
      new2 = (row.f2, row.f1)
      if (not new in existent) and (not new2 in existent):
        existent.append(new)
      else:
        to_remove.append(index)
    #print("to_remove: ", to_remove)
    df_feat = df_feat.drop(to_remove)
    df_feat = df_feat.reset_index(drop=True)
    #print(f"Pairs of aggregated features with corr > {corr_min}: {len(df_feat)}")
    col_to_remove=df_feat["f2"]
    for c in col_to_remove:
        if c in df.columns:
            df=df[df.columns[df.columns != c]]
    return df
