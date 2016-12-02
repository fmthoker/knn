import numpy as np
from scipy  import spatial
import scipy 
import sys
import random
from copy import copy
import heapq
class knn:
    
    def __init__(self):
        self.training_data=np.array([])
        self.labels=[]
        self.labelsTest=[]
        self.labelsresult=[]
        self.genes=[]
        self.test_data=np.array([])
        pass

    def normalize_training_data(self):
        print 'Normalising the training data'
        for i in range(0,len(self.training_data[0])):
		values = self.training_data[:,i]
		mean =  values.mean()
		std=  values.std()	
		for j in range(0,self.training_data.shape[0]):
		     self.training_data[j][i]= (self.training_data[j][i]- mean)/std
    
    def calculate_accuracy(self,gold,predicted):
        output=gold
        predic=predicted
        
        correct=0
        for i in range(0,len(output)):
            if output[i][0]==predic[i]:
                correct+=1
        
        return 100*(float(correct)/len(output))
        
                    
    def test(self):
        # Test for each row and predict output
        for i in range(0,len(self.labelsTest)):
            print 'testing ',self.labelsresult[i],' predicted= ',self.labels[i],' output is ',self.labelsTest[i]
            
            
        accuracy=self.calculate_accuracy(self.labelsTest,self.labelsresult)
        print 'Accuracy is ',accuracy,'%'
    def evaluate_k_neighbors(self):

      for k in [3,5]: # classification of data starts form this point
        self.labelsresult=[]
        for i in range(0,self.test_data.shape[0]):
		
                distances=[]
		for j in range(0,self.training_data.shape[0]):
		      dist= spatial.distance.cosine(self.training_data[j],self.test_data[i])
		      distances.append([dist])
		kmins= heapq.nsmallest(k,distances)
                neighbors=[]
		for z in kmins:
		       neighbors.append(distances.index(z))
                targets=[]
		for x in neighbors:
                       targets.append(self.labels[x])
                targets=np.array(targets)
                positives=np.where(targets==1)[0]
                negatives=np.where(targets==0)[0]
		if(len(positives) >=len(negatives)):
		       self.labelsresult.append(1)
                else:
		       self.labelsresult.append(0)
                del distances
		del neighbors
		del targets
	#test the results for whole test data for corresponding k value 
        self.test()
    
    def load_train_data(self):
        
        f=open(sys.argv[1])
        
        print 'Storing the training data'
        for row in f:        #Read input data as rows
           
            self.AllValues={}
            
            row=row.rstrip()   
            attributes=row.split(',')
            attributes_2=[float(i) for i in attributes[0:len(attributes)-1]]
            self.labels.append(float(attributes[-1]))
            
            attributes=copy(np.asarray(attributes_2))
            attributes=attributes.reshape(1,len(attributes))
            if self.training_data.shape[0]==0:
                self.training_data=copy(attributes)
            else:
                self.training_data=copy(np.vstack((self.training_data,attributes)))
                
        self.labels=copy(np.asarray(self.labels))
        self.labels=copy(self.labels.reshape(-1,1))
        self.normalize_training_data()
	

		
    

    def load_test_data(self):
        
         #open test file
        f=open(sys.argv[2])
        
        for row in f:
            row=row.rstrip()   
            row=row[0:len(row)-1]
            attributes=row.split(',')
            attributes_2=[float(i) for i in attributes[0:len(attributes)-1]]
            self.labelsTest.append(float(attributes[-1]))
            
            attributes=copy(np.asarray(attributes_2))
            attributes=attributes.reshape(1,len(attributes_2))
            if self.test_data.shape[0]==0:
                self.test_data=copy(attributes)
            else:
                self.test_data=copy(np.vstack((self.test_data,attributes)))
                
        self.labelsTest=copy(np.asarray(self.labelsTest))
        self.labelsTest=copy(self.labelsTest.reshape(-1,1))
        self.evaluate_k_neighbors()
        
    
if(len(sys.argv)!=3):
	print "Usage dt.py filename1 filename2"
    
else:
        
	OBJECT=knn()
	OBJECT.load_train_data()
	OBJECT.load_test_data()

