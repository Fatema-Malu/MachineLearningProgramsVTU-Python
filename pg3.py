
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 14:09:01 2019

@author: M205SYS001
"""

import pandas as pd
import numpy as np
dataset= pd.read_csv('playtennis.csv')
print ("dataset without names:")
print(dataset)
attributes=['outlook','temp','humidity','wind','class']

def entropy(target_col):
    elements,counts=np.unique(target_col,return_counts=True)
    #print("elements:",elements," counts:",counts)
    entropy=np.sum([(-counts[i]/np.sum(counts))* np.log2(counts[i]/np.sum(counts)) for i in range (len(elements))])
    #print("entropy:",entropy)
    return entropy

def infogain(data,split_attribute_name,target_name="class"):
    #print("data[target_name]:",data[target_name])
    total_entropy=entropy(data[target_name])
    vals,counts=np.unique(data[split_attribute_name],return_counts=True)
    #print("vals:",vals," counts:",counts)
    weighted_entropy=np.sum([(counts[i]/np.sum(counts))*entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range (len(vals))])
    information_gain= total_entropy - weighted_entropy
    return information_gain

def ID3 (data,original_data,features, target_attribute_name="class", parent_node_class=None):
    if len(np.unique(data[target_attribute_name]))<=1:
        return np.unique(data[target_attribute_name])[0]
    elif len(features)==0:
        return parent_node_class
    else:
        unique_TA= np.unique(data[target_attribute_name], return_counts=True)[1]
        #print("unique TA:",unique_TA)
        max_unique_TA=np.argmax(unique_TA)
        parent_node_class=np.unique(data[target_attribute_name])[max_unique_TA]
        #print("parent node class:", parent_node_class)
        item_values=[infogain(data,feature,target_attribute_name) for feature in features]
        #print("item values:",item_values)
        best_feature_index=np.argmax(item_values)
        best_feature=features[best_feature_index]
        tree={best_feature:{}}
        features=[i for i in features if i!= best_feature]
        for value in np.unique(data[best_feature]):
            value=value
            sub_data=data.where(data[best_feature]==value).dropna()
            #print("sub data:", sub_data)
            subtree=ID3(sub_data, dataset, features, target_attribute_name, parent_node_class)
            tree[best_feature][value]=subtree
    return (tree)

def predict(query,tree):
    for key in list(query.keys()):
        if key in list(tree.keys()):
            result=tree[key][query[key]]
    if isinstance(result,dict):
        return predict(query,result)
    else:
        return result

def train_test_split(dataset):
    training_data=dataset.iloc[:14]
    return training_data

def test(data,tree):
    queries=data.iloc[:,:-1].to_dict(orient="records")
    predicted=pd.DataFrame(columns=["predicted"])
    for i in range (len(data)):
        predicted.loc[i,"predicted"]=predict(queries[i],tree)
    #print("Predicted : ",predicted)
    print('the prediction accuracy is:',(np.sum(predicted["predicted"]==data["class"])/len(data))*100,'%')
   
xx=train_test_split(dataset)
training_data=xx
tree=ID3(training_data, training_data, training_data.columns[:-1])
print('display tree',tree)
#print('len=',len(training_data))
test_data=dataset.iloc[14:17].reset_index(drop=True)
test(test_data,tree)