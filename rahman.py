import sys
import numpy as np
import pandas as pd
from itertools import combinations, chain
import time

start = time.time()

print("Welcome to the simulation of the apriori algorithms. \n Please choose the dataSet you want: \n 1. Db1 \n 2. Db2 \n 3. Db3 \n 4. Db4 \n 5. Db5")
while True:
    choice_of_data=input()
    if(choice_of_data=='1'):
        dataList=pd.read_csv('db1.txt')
        print('User chose db1 dataset')
        break
    elif(choice_of_data=='2'):
        dataList=pd.read_csv('db2.txt')
        print('User chose db2 dataset')
        break
    elif(choice_of_data=='3'):
        dataList=pd.read_csv('db3.txt')
        print('User chose db3 dataset')
        break
    elif(choice_of_data=='4'):
        dataList=pd.read_csv('db4.txt')
        print('User chose db4 dataset')
        break
    elif(choice_of_data=='5'):
        dataList=pd.read_csv('db5.txt')
        print('User chose db5 dataset')
        break
    else:
        print("Invalid data, please enter the number corresponding to the data")
        break     
    

print("Enter the minimum support (in percentage) : ", end=" ")
minsupport = input()
print("Enter the minimum confidence (in percentage) : ", end=" ")
minconfidence = input()

min_support =  int(minsupport)/100
min_confidence = int(minconfidence)/100


#def load_transactions(path_to_data):
#    Transactions = []
#    with open(path_to_data, 'r') as fid:
#        for lines in fid:
#            str_line = list(lines.strip().split(','))
#            _t = list(np.unique(str_line))
#            ##_t.sort(keys = lambda x: order.index(x))
#            Transactions.append(_t)
#    return Transactions
def load_transactions(dataList):
    Transactions = []
    df_items = dataList['TransactionList']
    comma_splitted_df = df_items.apply(lambda x: x.split(','))
    for i in comma_splitted_df:
        Transactions.append(i)
    return Transactions
load_transactions(dataList)

Transactions = load_transactions(dataList)
Transactions   
    
#path_to_data = sys.argv[1]

#Transactions = load_transactions(path_to_data)

#print(transactions);

C = {}
L = {}
itemset_size = 1
discarded = { itemset_size : [] }

##C.update(itemset_size)
l1= []

for i in Transactions:
    for j in i:
        if j not in l1:
            l1.append(j)
            
l2 = np.reshape(l1,(len(l1),1))
l3 = l2.tolist()

            

#print("C: -----");



C.update({itemset_size : l3})
#print(C)

def count_occurences(itemset, Transactions):
    count = 0
    for i in range(len(Transactions)):
        if set(itemset).issubset(set(Transactions[i])):
            count += 1
    return count

def get_frequent(itemsets, Transactions, min_support, prev_discarded):
    L = []
    supp_count = []
    new_discarded = []
        
    k = len(prev_discarded.keys())
    
    for s in range(len(itemsets)):
        discarded_before = False
        if k>0:
            for it in prev_discarded[k]:
                if set(it).issubset(set(itemsets[s])):
                    discarded_before = True
                    break
         
        if not discarded_before:
            count = count_occurences(itemsets[s], Transactions)
            if count/len(Transactions) >= min_support:
                L.append(itemsets[s])
                supp_count.append(count)
            else:
                new_discarded.append(itemsets[s])
    
    return L, supp_count, new_discarded
    
supp_count_L = {}
f , sup ,  new_discarded = get_frequent(C[itemset_size], Transactions, min_support, discarded)
discarded.update({itemset_size : new_discarded})
L.update({itemset_size :  f})
supp_count_L.update({itemset_size : sup})
#print("L1: \n")
##print(tabulate([L[1], supp_count_L[1]]))
##print(L[1] , supp_count_L[1])
#df = pd.DataFrame({L[1] : supp_count_L[1]}.items() , columns = ["Items" , "Frequency"])
#print(df)
#for i in range(len(L[1])):
#   print( L[1][i] , supp_count_L[1][i])
    

def join_two_itemsets(it1, it2, l1):
    it1.sort(key = lambda x: l1.index(x))
    it2.sort(key = lambda x: l1.index(x))
    
    for i in range( len(it1) - 1):
        if it1[i] != it2[i]:
            return []
            
    if l1.index(it1[-1]) < l1.index(it2[-1]):
        return it1 + [it2[-1]]
        
    return []

def join_set_itemsets(set_of_its, l1):
    C = [] 
    for i in range(len(set_of_its)):
        for j in range(i+1, len(set_of_its)):
            it_out = join_two_itemsets(set_of_its[i], set_of_its[j], l1)
            if len(it_out) > 0:
                C.append(it_out)
    return C     



k = itemset_size + 1
convergence = False

while not convergence:
    C.update({k : join_set_itemsets(L[k-1], l1)})
    #print("Table C{}: \n".format(k))
    #print(tabulate(C[k], [count_occurences(it, Transactions) for it in C[k]]))
    f, sup, new_discarded = get_frequent(C[k], Transactions, min_support, discarded)
    discarded.update({k : new_discarded})
    L.update({k : f })
    supp_count_L.update({k : sup})
    if len(L[k]) == 0:
        convergence = True
    #else:
        #print("Table L{}: \n".format(k))
        #print(tabulate(L[k], supp_count_L[k]))
    k += 1
    
def powerset(s):
    return list(chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1)))
    
def write_rules(X, X_S, S, conf, supp, lift, num_trans):
    out_rules = ""
    #out_rules += " Freq. Itemset: {}\n".format(X)
    out_rules += "    Rule {} -> {}\n".format(list(S), list(X_S))
    out_rules += "    Conf: {0:2.3f}".format(conf)
    out_rules += "    Supp: {0:2.3f}".format(supp/num_trans)
    out_rules += "    Lift: {0:2.3f}\n".format(lift)
    return out_rules
    
    
assoc_rules_str =""

for i in range(1, len(L)):
    for j in range(len(L[i])):
        s = powerset(L[i][j])
        s.pop()
        for z in s: 
            S = set(z)
            X = set(L[i][j])
            X_S = set(X-S)
            sup_x = count_occurences(X, Transactions)
            sup_x_s = count_occurences(X_S, Transactions)
            conf = sup_x / count_occurences(S, Transactions)
            lift = conf / (sup_x_s / len(Transactions))
            if conf >= min_confidence and sup_x >= min_support:
                assoc_rules_str += write_rules(X, X_S, S, conf, sup_x, lift, len(Transactions))
                

print (assoc_rules_str)            

end = time.time()

print(f"Run time of the program is {end - start}")
