import time

def prune(item,freq_itemsets):
    for i in range(len(item)):
        skip = item[:i]+item[i+1:]
        if skip not in freq_itemsets:
            return False
    return True

def aprioriGen(freq_itemsets):
    nextCandidates=[]
    for i in range(len(freq_itemsets)):
        for j in range(i+1,len(freq_itemsets)):
            lst = len(freq_itemsets[i])
            temp = []
            flag = True
            for k in range(lst-1):
                if freq_itemsets[i][k]==freq_itemsets[j][k]:
                    temp.append(freq_itemsets[i][k])
                else:
                    flag = False
                    break
            if freq_itemsets[i][-1]>=freq_itemsets[j][-1]:
                flag = False
            temp = temp + [freq_itemsets[i][-1]] + [freq_itemsets[j][-1]]
            if flag and prune(temp, freq_itemsets):
                nextCandidates.append(temp)
    return nextCandidates


def hash_for_2_candidates(transactions, freq_itemsets, minSupport):
    
    new_items = []
    counter = dict()
    
    for i in transactions:
        freq_t = []
        for j in i:
            if [j] in freq_itemsets:
                freq_t.append(j)
        for it1 in range(len(freq_t)):
            for it2 in range(len(freq_t)):
                if freq_t[it2]>freq_t[it1]:
                    if (freq_t[it1], freq_t[it2]) not in counter:
                        counter[(freq_t[it1], freq_t[it2])]=1
                    else:
                        counter[(freq_t[it1], freq_t[it2])] = counter[(freq_t[it1], freq_t[it2])] + 1

    updated_items = []
    
    for i in counter:
        if counter[i]>=minSupport:
            updated_items.append([i[0], i[1]])
    
    return updated_items
    
    
def update_items(dataset, candidates, minsup):


    counter = []
    future_transactions = []
    for i in range(len(candidates)):
        counter.append(0)
        future_transactions.append([])



    for i in range(len(dataset)):
        for j in range(len(candidates)):
            if set(candidates[j]) <= set(dataset[i]):
                counter[j]+=1
                future_transactions[j].append(i)
    updated_items = []
    for i in range(len(candidates)):
        if counter[i]>=minsup:
            updated_items.append(candidates[i])
    return updated_items

def _all_one_itemset(data):
    ans = []
    for i in data:
        ans = ans + i
    ans = list(set(ans))
    ans = [[i] for i in ans]
    return ans

def getData(filename):
    with open(filename, 'r') as file:
        data = file.read().replace('\n', '')
    file.close()
    data = data.split('-2')
    for i in range(len(data)):
        data[i] = data[i].split('-1')
        data[i] = data[i][:-1]
        data[i] = list(map(int, data[i]))
    sm=0
    for i in data: sm = sm + len(i)
    print(f'\nThe dataset has {len(data)} transactions with average {sm/len(data)} items per transaction.')
    return data

def update_items_from_partitions(dataset, candidates, minsup, partition_count):
    
    starting_index = 0
    ans = []
    minsup = minsup // partition_count
    
    tt = []
    
    for i in range(partition_count):
        
        lst = time.time()
        
        it1 = starting_index
        it2 = starting_index + len(dataset)//partition_count-1
        starting_index = it2 + 1
        
        counter = []
        future_transactions = []
        
        for i in range(len(candidates)):
            counter.append(0)
            future_transactions.append([])
        
        for i in range(it1, it2+1):
            for j in range(len(candidates)):
                if set(candidates[j]) <= set(dataset[i]):
                    counter[j]+=1
                    future_transactions[j].append(i)
                    
        updated_items = []

        for i in range(len(candidates)):
            if counter[i]>=minsup:
                updated_items.append(candidates[i])
                
        tt.append(time.time()-lst)
        ans = ans + updated_items
        new_ans = []
        for elem in ans:
            if elem not in new_ans:
                new_ans.append(elem)
        ans = new_ans.copy()

    return ans, max(tt)

if __name__=="__main__":
    
    partition_count = 6
    dataset = getData('SIGN.txt')
    all_one_itemset = _all_one_itemset(dataset)
#     _minsup = [600, 550, 500, 450]
    _minsup = [400]

    for minsup in _minsup:

        print(f'\nMINSUP: {minsup}')

        tot_time = 0

        frequent_itemset = dict()
        lst = time.time()
        frequent_itemset[1] = update_items(dataset, all_one_itemset, minsup)
        frequent_itemset[2] = hash_for_2_candidates(dataset, frequent_itemset[1], minsup)
        tot_time = tot_time + time.time()-lst
        candidates = frequent_itemset[2]
        
        k=2
        while candidates:
            candidates, tt = update_items_from_partitions(dataset, candidates, minsup, partition_count)
            tot_time = tot_time + tt
            
            lst = time.time()
            frequent_itemset[k] = update_items(dataset, candidates, minsup)
            candidates = aprioriGen(frequent_itemset[k])
            tot_time = tot_time + time.time()-lst
            k = k+1
        
        for i in frequent_itemset:
            print('NUMBER OF ITEMS: ', i)
            print(frequent_itemset[i])
        
        print(f'TIME: {tot_time} seconds')

