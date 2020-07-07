# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 21:09:38 2019

@author: HPO
"""

import csv
import math

def sd(x,y):
    if y==0:
        return 0
    return x/y

def loadcsv(filename):
    lines=csv.reader(open(filename))
    dataset=list(lines)
    for i in range(len(dataset)):
        dataset[i]=[float(x) for x in dataset[i]]
    return dataset

def splitdataset(dataset,splitratio):
    trainsize=int(len(dataset)*splitratio)
    trainset=[]
    copy=list(dataset)
    i=0
    while len(trainset)<trainsize:
        trainset.append(copy.pop(i))
    return [trainset,copy]

def seperatebyclass(dataset):
    sep={}
    for i in range(len(dataset)):
        vector=dataset[i]
        if(vector[-1] not in sep):
            sep[vector[-1]].append(vector)
    return sep

def mean(num):
    return sd(sum(num),float(len(num)))

def stdev(num):
	avg = mean(num)		
	variance = sd(sum([pow(x-avg,2) for x in num]),float(len(num)-1))
	return math.sqrt(variance)
    
def sumerize(dataset):
    sumer=[(mean(att),stdev(att)) for att in zip(*dataset)]
    del sumer[-1]
    return sumer

def sumerisebyclass(dataset):
    sep=seperatebyclass(dataset)
    sumerise={}
    for classvalue,ins in sep.items():
        sumerise[classvalue]=sumerize(ins)   
    return sumerise

def clapob(x,mean,stdev):
    exponent=math.exp(-sd(math.pow(x-mean,2),(2*math.pow(stdev,2))))
    final=sd(1,math.sqrt(2*math.pi,stdev))*exponent
    return final
    
    
    
    