# -*- coding: utf-8 -*-
"""
Created on Thu May 30 10:26:14 2019

@author: sachi
"""

# A Python program for Prim's Minimum Spanning Tree (MST) algorithm. 
# The program is for adjacency matrix representation of the graph 

import sys # Library for INT_MAX 
import numpy as np
import csv
from docplex.mp.model import Model
from docplex.mp.context import Context
import datetime
import networkx as nx
import CATMNetworkGraphINT

#Start of the MST using Prims Algorithm
class Graph(): 
  
    def __init__(self, vertices): 
        self.V = vertices 
        self.graph = [[0 for column in range(vertices)]  
                    for row in range(vertices)] 
  
    # A utility function to print the constructed MST stored in parent[] 
    def printMST(self, parent): 
        print ("Edge \tWeight")
        for i in range(1,self.V): 
            print (parent[i],"-",i,"\t",self.graph[i][ parent[i] ]) 
  
    # A utility function to find the vertex with  
    # minimum distance value, from the set of vertices  
    # not yet included in shortest path tree 
    def minKey(self, key, mstSet): 
  
        # Initilaize min value 
        min = sys.maxsize 
  
        for v in range(self.V): 
            if key[v] < min and mstSet[v] == False: 
                min = key[v] 
                min_index = v 
  
        return min_index 
  
    # Function to construct and print MST for a graph  
    # represented using adjacency matrix representation 
    def primMST(self): 
  
        #Key values used to pick minimum weight edge in cut 
        key = [sys.maxsize] * self.V 
        parent = [None] * self.V # Array to store constructed MST 
        # Make key 0 so that this vertex is picked as first vertex 
        key[0] = 0 
        mstSet = [False] * self.V 
  
        parent[0] = -1 # First node is always the root of 
  
        for cout in range(self.V): 
  
            # Pick the minimum distance vertex from  
            # the set of vertices not yet processed.  
            # u is always equal to src in first iteration 
            u = self.minKey(key, mstSet) 
  
            # Put the minimum distance vertex in  
            # the shortest path tree 
            mstSet[u] = True
  
            # Update dist value of the adjacent vertices  
            # of the picked vertex only if the current  
            # distance is greater than new distance and 
            # the vertex in not in the shotest path tree 
            for v in range(self.V): 
                # graph[u][v] is non zero only for adjacent vertices of m 
                # mstSet[v] is false for vertices not yet included in MST 
                # Update the key only if graph[u][v] is smaller than key[v] 
                if self.graph[u][v] > 0 and mstSet[v] == False and key[v] > self.graph[u][v]: 
                        key[v] = self.graph[u][v] 
                        parent[v] = u 
  
        self.printMST(parent)
        return parent

#End of MST 

#Input files
InputFile = "4FED3P70"
STRROutput1 = InputFile + "STRR1"
LTRROutput1 = InputFile + "LTRR1"
STRROutput2 = InputFile + "STRR2"
LTRROutput2 = InputFile + "LTRR2"

allEdges = np.genfromtxt(InputFile +".csv", dtype='int', delimiter=',')
edgeList = allEdges.tolist()
g = Graph(50)
for edge in edgeList:
    if edge[3] == 0:
        g.graph[edge[0]-1][edge[1]-1] = 1
        g.graph[edge[1]-1][edge[0]-1] = 1
    else:
        g.graph[edge[0]-1][edge[1]-1]=edge[3]
        g.graph[edge[1]-1][edge[0]-1]= edge[3]

mst = g.primMST()
for i in range(1,g.V):
    if g.graph[i][mst[i]] > 1:
        
        for j in range(0,len(edgeList)):
            if (i==(edgeList[j][0]-1) and mst[i]==(edgeList[j][1]-1)) or (i==(edgeList[j][1]-1) and mst[i]==(edgeList[j][0]-1)):
                edgeList[j][2] = 1
                break
MSTendTime = datetime.datetime.now()   

G=nx.Graph()
    # add edges
for edge in edgeList :
    G.add_edge(edge[0], edge[1], status=edge[2])		                  

nodeGroup =  CATMNetworkGraphINT.Nodes_Cluster(G,46)
#print(nodeGroup)
#g = Graph(2)
#g.graph = np.genfromtxt("NCCountyLinks1.csv", dtype='int', delimiter=',')

startTime = datetime.datetime.now()


#with open("fileForRP.csv",'w',newline ='') as outputFile:
#    csv_writer = csv.writer(outputFile)
#    
#    for line in edgeList:
#        csv_writer.writerow(line)
        

#for Short-term restoration (STRR)
mqSTRR = Model(name="STRR")

#reading csv files

#allEdges = np.genfromtxt("fileForRP.csv", dtype='int', delimiter=',')
#edgeList = allEdges.tolist()

AllEdges = []
for edge in edgeList:
    if edge[2] <2:
        AllEdges.append((edge[0],edge[1]))
        
DamageEdges = []
for edge in edgeList:
    if edge[2] > 0:
        DamageEdges.append((edge[0],edge[1]))
EdgeMST = []
workloadofMST = 0
for edge in edgeList:
    if edge[2]==1:
        EdgeMST.append((edge[0], edge[1]))
        workloadofMST += edge[3]

# Setting of the model
DailyCapacity = 1664
numDays = int(workloadofMST/DailyCapacity)+1
totalHours = 24*numDays
numTimePeriods = numDays*3   
numNodes = 50
weightRCC = 1
weightAirport = 1

#CountynPopulation = np.genfromtxt("INTCountyPopulation.csv", delimiter=",", dtype=None, encoding= None)
#my_dict = dict()
#for i in range(len(CountynPopulation)):
#   my_dict[CountynPopulation[i][0]] = CountynPopulation[i][1]

CountynPopulation = {}
with open('INTCountyPopulation.csv','r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:  
         CountynPopulation[(row[0],row[1])] = row[2]
         
#Sets
AllNodes = list(range(1,51))
SourceNodes = [46]
DemandNodes = AllNodes.copy()
for i in SourceNodes:
    DemandNodes.remove(i)
TimePeriods = list(range(1,numTimePeriods+1))

Edgelist = AllEdges
DmgEdge = EdgeMST

#parameters ( dictionaries from csv files)
PopulationInCounty = {} 
for key, value in CountynPopulation.items(): 
   PopulationInCounty[(int(key[0]),46)] =int (value)

#Parameter of workload for edges retriving from edgelist of MST 
WorkloadEdge ={}
for edge in edgeList:
    WorkloadEdge[(edge[0],edge[1])] = edge[3] 

#Assigning float value to workload
EdgeWorkload = {}
for key, value in WorkloadEdge.items(): 
   EdgeWorkload[(int(key[0]),int(key[1]))] = float (value)
   
#Decision variables
#flow from i to j
f = {(e[0],e[1],t) : mqSTRR.continuous_var(name = "f_e{0}_{1}_t{2}".format(e[0],e[1],t)) 
                    for e in Edgelist for t in TimePeriods}
for e in Edgelist:
    for t in TimePeriods:
        f[(e[1],e[0],t)] = mqSTRR.continuous_var(name = "f_e{0}_{1}_t{2}".format(e[1],e[0],t)) 

#path from demand to source node
z = {(d,s,t) : mqSTRR.binary_var(name = "z_d{0}_s{1}_t{2}".format(d,s,t)) 
for s in SourceNodes for d in DemandNodes for t in TimePeriods}

#gamma in the model
g = {(e,t) : mqSTRR.binary_var(name = "g_dmgedge{0}_{1}_t{2}".format(e[0],e[1],t)) 
for e in DmgEdge for t in TimePeriods} 

# Y cumulative resource allocated
YC = {(e,t) : mqSTRR.continuous_var(name = "YC_dmgedge{0}_{1}_t{2}".format(e[0],e[1],t)) 
for e in DmgEdge for t in TimePeriods}

# small y in model 
y = {(e,t) : mqSTRR.continuous_var(name = "y_dmgedge{0}_{1}_t{2}".format(e[0],e[1],t)) 
for e in DmgEdge  for t in TimePeriods}

#objective function

mqSTRR.maximize(mqSTRR.sum(PopulationInCounty.get((d,s),0)* z[d,s,t] 
for d in DemandNodes for s in SourceNodes for t in TimePeriods))

#Constraints        
for e in DmgEdge:
    for t in TimePeriods:
        mqSTRR.add_constraint(100*g[e,t] >= f[e[0],e[1],t])
        mqSTRR.add_constraint(100*g[e,t] >= f[e[1],e[0],t])
          

for s in SourceNodes:
    DNodes = set()
    for e in Edgelist:
        if e[0] == s:
            DNodes = DNodes.union({e[1]})
        if e[1] == s:
            DNodes = DNodes.union({e[0]})
#    print(DNodes)
    for t in TimePeriods:
#        print(mqSTRR.sum(z[d,s,t] for d in DemandNodes)
#            + mqSTRR.sum(f[k,s,t] for k in DNodes) == mqSTRR.sum (f[s,l,t] for l in DNodes))    
        mqSTRR.add_constraint(mqSTRR.sum(z[d,s,t] for d in DemandNodes)
            + mqSTRR.sum(f[k,s,t] for k in DNodes) == mqSTRR.sum (f[s,l,t] for l in DNodes))    
                                    
#for demand node
for d in DemandNodes:
    DNodes = set()
    for e in Edgelist:
        if e[0] == d:
            DNodes = DNodes.union({e[1]})
        if e[1] == d:
            DNodes = DNodes.union({e[0]})
#    print(DNodes)
    for t in TimePeriods:
        mqSTRR.add_constraint(mqSTRR.sum(f[k,d,t] for k in DNodes )
        == mqSTRR.sum(z[d,s,t] for s in SourceNodes) + mqSTRR.sum(f[d,l,t] for l in DNodes))
                
#workload
for e in DmgEdge:
    for t in TimePeriods:
        # w1
#        print(YC[e,t] <= EdgeWorkload.get(e,0) * g[e,t])
        mqSTRR.add_constraint(YC[e,t] >= EdgeWorkload.get(e,0) * g[e,t])
        # w2
        mqSTRR.add_constraint(EdgeWorkload.get(e,0) - YC[e,t] >= 1 - g[e,t])
        
        #Clearance Cumulative
        if t == 1:
#            print(YC[e,t] == y[e,t])
            mqSTRR.add_constraint(YC[e,t] == y[e,t])
        else:
#            print(YC[e,t] == YC[e,t-1] + y[e,t])
            mqSTRR.add_constraint(YC[e,t] == YC[e,t-1] + y[e,t])
        
for t in TimePeriods:
    mqSTRR.add_constraint(mqSTRR.sum(y[e,t] for e in DmgEdge) <= 554)
                
# Connectivity at the last time period
#for d in DemandNodes:
#    for s in SourceNodes:
#        t = 9
#        mqSTRR.add_constraint(z[d,s,t] == 1 )
#            print(mqSTRR.add_constraint(z[d,s,t])== 1)
         
#solution
#print(mqSTRR.get_objective_expr())
print('Solving the STRR problem...')
STRRstartTime = datetime.datetime.now()
mqSTRR.solve()
STRRendTime = datetime.datetime.now()
#mqSTRR.print_solution() 
print('The computational time of solving the MST problem is: ', MSTendTime - startTime, '\n')
print('The computational time of solving the STRR problem is: ', STRRendTime - STRRstartTime, '\n')
#mqSTRR.solve_details

#### Results of the STRR Problem
# Road restoration sequence based on gamma in the model
dictRoadResSequence = {}
listRestorationSequence1 = []
# Resource allocation (smalll y in model) 
dictResAllocation = {}
#g = {(e,t) : mqSTRR.binary_var(name = ")
for e in DmgEdge: 
    for t in TimePeriods:
        nameRoad = "g_dmgedge{0}_{1}_t{2}".format(e[0],e[1],t)
        var = int(mqSTRR.get_var_by_name(nameRoad).solution_value)
        if (var > 0) and (dictRoadResSequence.get(e) == None):
            dictRoadResSequence[e] = t
            listRestorationSequence1.append([t,e])

        nameResource = "y_dmgedge{0}_{1}_t{2}".format(e[0],e[1],t)
        var = round(mqSTRR.get_var_by_name(nameResource).solution_value)
        if var > 0:
            dictResAllocation[(e,t)] = var

listRestorationSequence1.sort()

#writing csv file for road restoration sequence
with open( STRROutput1+".csv",'w',newline ='') as outputFile:
    csv_writer = csv.writer(outputFile)
    
    for road, timeperiod in dictRoadResSequence.items():
        csv_writer.writerow([road, timeperiod])           
        
DictPathSD ={}
for d in nodeGroup[0]:
#    print(d)
    if d != 46:
        DictPathSD[d]=0
#listPathSD = []
for s in SourceNodes:
    for d in DemandNodes:
        for t in TimePeriods:
            VarName = "z_d{0}_s{1}_t{2}".format(d,s,t)
            value =int(mqSTRR.get_var_by_name(VarName).solution_value)
            if (value > 0) and (DictPathSD.get(d) == None):
                DictPathSD[d] = t  
#                listPathSD.append([t,d])
#listPathSD.sort()

##writing csv file for Z with population of county
with open(STRROutput2+".csv",'w',newline ='') as outputFile:
    csv_writer = csv.writer(outputFile)
    
    for demandNode, timePeriod in DictPathSD.items():
        csv_writer.writerow([demandNode,timePeriod, PopulationInCounty[(demandNode,46)]]) 

 
#for LTRR Long term restoration
ct = Context.make_default_context()
ct.cplex_parameters.mip.tolerances.mipgap = 0.01
mqLTRR= Model(name="LTRR", context = ct)

## Settins of the model
#numDays = 30
#totalHours = 24*numDays
numTimePeriods = 40    
#numNodes = 50
#weightRCC = 1
#weightAirport = 1

#reading csv files

#allEdges = np.genfromtxt("fileForRP.csv", dtype='int', delimiter=',')
#edgeList = allEdges.tolist()

AllEdges = []
for edge in edgeList:
    AllEdges.append((edge[0],edge[1]))
        
DamageEdges = []
for edge in edgeList:
    if edge[2] > 1:
        DamageEdges.append((edge[0],edge[1]))
        
AADTforRoads = {}
WorkloadEdge ={}
for edge in edgeList:
    AADTforRoads[(edge[0],edge[1])] = edge[5]
    WorkloadEdge[(edge[0],edge[1])] = edge[3] 


#Sets
AllNodes = list(range(1,51))
SourceNodes = [46]
DemandNodes = AllNodes.copy()
for i in SourceNodes:
    DemandNodes.remove(i)
TimePeriods = list(range(1,numTimePeriods+1))

Edgelist = AllEdges

#gamma in the model
g = {(e,t) : mqLTRR.binary_var(name = "g_dmgedge{0}_{1}_t{2}".format(e[0],e[1],t)) 
for e in DamageEdges for t in TimePeriods} 

# Y cumulative resource allocated
YC = {(e,t) : mqLTRR.continuous_var(name = "YC_dmgedge{0}_{1}_t{2}".format(e[0],e[1],t)) 
for e in DamageEdges for t in TimePeriods}

# small y in model 
y = {(e,t) : mqLTRR.continuous_var(name = "y_dmgedge{0}_{1}_t{2}".format(e[0],e[1],t)) 
for e in DamageEdges  for t in TimePeriods}


#objective function
mqLTRR.maximize(mqLTRR.sum(AADTforRoads.get(e,0)*g[e,t] 
for e in DamageEdges for t in TimePeriods))
          
#workload
for e in DamageEdges:
    for t in TimePeriods:
        # w1
#        print(YC[e,t] <= EdgeWorkload.get(e,0) * g[e,t])
        mqLTRR.add_constraint(YC[e,t] >= WorkloadEdge.get(e,0) * g[e,t])
        # w2
        mqLTRR.add_constraint(WorkloadEdge.get(e,0) - YC[e,t] >= 1 - g[e,t])
        
        #Clearance Cumulative
        if t == 1:
#            print(YC[e,t] == y[e,t])
            mqLTRR.add_constraint(YC[e,t] == y[e,t])
        else:
#            print(YC[e,t] == YC[e,t-1] + y[e,t])
            mqLTRR.add_constraint(YC[e,t] == YC[e,t-1] + y[e,t])
            mqLTRR.add_constraint(g[e,t] >= g[e,t-1])
        
for t in TimePeriods:
    mqLTRR.add_constraint(mqLTRR.sum(y[e,t] for e in DamageEdges) <=1664)
                
#solution
#print(mqLTRR.get_objective_expr())
print('Solving the LTRR problem...')
LTRRstartTime = datetime.datetime.now()
mqLTRR.solve()
LTRRendTime = datetime.datetime.now()
#mqLTRR.print_solution() 

#mqLTRR.solve_details
print('The computational time of solving the LTRR problem is: ', LTRRendTime - LTRRstartTime, '\n')
endTime = datetime.datetime.now()
print('The computational time of solving the 3 problems is: ', endTime - startTime, '\n')

f = open(InputFile +"output.txt", "a")
print("The computational time of solving the MST problem is: ", MSTendTime - startTime, file =f)
print('The computational time of solving the STRR problem is: ', STRRendTime - STRRstartTime, file =f)
print("The computational time of solving the LTRR problem is: ", LTRRendTime - LTRRstartTime, file=f)
print("The computational time of solving the 3 problems is: ", endTime - startTime, file=f)
f.close()

### Results of the LTRR problem
# Road restoration sequence based on gamma in the model
dictRoadResSequence = {}
listRestorationSequence2 = []
# Resource allocation (smalll y in model) 
dictResAllocation = {}
#g = {(e,t) : mqLTRR.binary_var(name = ") 
for e in DamageEdges: 
    for t in TimePeriods:
        nameRoad = "g_dmgedge{0}_{1}_t{2}".format(e[0],e[1],t)
        var = int(mqLTRR.get_var_by_name(nameRoad).solution_value)
        if (var > 0) and (dictRoadResSequence.get(e) == None):
            dictRoadResSequence[e] = t
            listRestorationSequence2.append([t,e])

        nameResource = "y_dmgedge{0}_{1}_t{2}".format(e[0],e[1],t)
        var = round(mqLTRR.get_var_by_name(nameResource).solution_value)
        if var > 0:
            dictResAllocation[(e,t)] = var

listRestorationSequence2.sort()


#writing csv file
with open(LTRROutput1+ ".csv",'w',newline ='') as outputFile:
    csv_writer = csv.writer(outputFile)

    for road, timeperiod in dictRoadResSequence.items():
        csv_writer.writerow([road, timeperiod])

#writing csv file for road restoration sequence
with open(LTRROutput2+".csv",'w',newline ='') as outputFile:
    csv_writer = csv.writer(outputFile)
    
    for road, timeperiod in dictRoadResSequence.items():
        csv_writer.writerow([road, timeperiod,AADTforRoads.get(road)])  
        

        