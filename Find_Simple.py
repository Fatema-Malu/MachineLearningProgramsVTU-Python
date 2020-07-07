import csv

def loadCsv(filename):
    lines = csv.reader(open(filename))
    dataset = list(lines)
    print("Dataset : ",dataset)
    return dataset

attributes = ['Sky','Temp','Humidity','Wind','Water','Forecast']
print(attributes)
num_attributes = len(attributes)
print("NUM",num_attributes)

filename = "weat.csv"
dataset = loadCsv(filename)

target=['Yes','Yes','No','Yes']
print(target)

hypothesis=['0'] * num_attributes
print(hypothesis)

print("The Hypothesis are")
for i in range(len(target)):
    
    if(target[i] == 'Yes'):
        for j in range(num_attributes):        
            if(hypothesis[j]=='0'):
                hypothesis[j] = dataset[i][j]
            if(hypothesis[j]!= dataset[i][j]):
                hypothesis[j]='?'
      
    print(i+1,'=',hypothesis)
    
print("Final Hypothesis")
print(hypothesis)







