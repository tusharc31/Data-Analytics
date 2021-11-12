impot time

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
                if freq_t[it1]<freq_t[it2]:
                    if (freq_t[it1], freq_t[it2]) in counter:
                        counter[(freq_t[it1], freq_t[it2])] = counter[(freq_t[it1], freq_t[it2])] + 1
                    else:
                        counter[(freq_t[it1], freq_t[it2])]=1

    updated_items = []
    
    for i in counter:
        if counter[i]>=minSupport:
            updated_items.append([i[0], i[1]])
    
    return updated_items
    
    
def update_items(dataset, candidates, minsup, mark):
    counter = [0 for i in candidates]
    future_transactions = [[] for i in candidates]
    for i in range(len(dataset)):
        if mark[i]:
            for j in range(len(candidates)):
                if set(candidates[j]) <= set(dataset[i]):
                    counter[j]+=1
                    future_transactions[j].append(i)
    new_mark = [False for i in dataset]
    updated_items = []
    for i in range(len(candidates)):
        if counter[i]>=minsup:
            updated_items.append(candidates[i])
            for j in future_transactions[i]:
                new_mark[j] = True
    return updated_items, new_mark

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


if __name__=="__main__":
    
    dataset = getData('SIGN.txt')
    all_one_itemset = _all_one_itemset(dataset)
    _minsup = [400]

    for minsup in _minsup:

        print(f'\nMINSUP: {minsup}')
        start_time = time.time()

        mark = []

        for i in range(len(dataset)):
            mark.append(True)

        frequent_itemset = dict()
        frequent_itemset[1], mark = update_items(dataset, all_one_itemset, minsup, mark)
        frequent_itemset[2] = hash_for_2_candidates(dataset, frequent_itemset[1], minsup)
        candidates = frequent_itemset[2]
        
        k=2
        while candidates:
            frequent_itemset[k], mark = update_items(dataset, candidates, minsup, mark)
            candidates = aprioriGen(frequent_itemset[k])
            k = k+1
        
        for i in frequent_itemset:
            print('NUMBER OF ITEMS: ', i)
            print(frequent_itemset[i])
        
        print(f'TIME: {time.time()-start_time} seconds')
r
