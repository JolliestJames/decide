import pickle
import collections
import math
import operator

with open("data", "rb") as f:
    L = pickle.load(f)
    print(len(L))

def calculate_category_frequency(data):
    return collections.Counter([item[-1] for item in data])

def entropy(data):
    category_frequencies = calculate_category_frequency(data)

    def item_entropy(category):
        category_ratio = float(category)/len(data)
        return -1 * category_ratio * math.log(category_ratio, 2)
    
    return sum(item_entropy(category) for category in category_frequencies.values())

def best_feature_to_split(data):
    baseline_entropy = entropy(data)

    def feature_entropy(f):
        def e(v):
            partitioned_data = [d for d in data if d[f] == v]
            proportion = (float(len(partitioned_data))/float(len(data)))
            return proportion * entropy(partitioned_data)
        return sum(e(v) for v in set([d[f] for d in data]))

    features = len(data[0]) - 1

    information_gain = [baseline_entropy - feature_entropy(f) for f in range(features)]

    best_feature, best_gain = max(enumerate(information_gain), key = operator.itemgetter(1))

    return best_feature
