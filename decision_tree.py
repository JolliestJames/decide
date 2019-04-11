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
            partitioned = [d for d in data if d[f] == v]
            proportion = (float(len(partitioned))/float(len(data)))
            return proportion * entropy(partitioned)
        return sum(e(v) for v in set([d[f] for d in data]))

    feature_count = len(data[0]) - 1

    information_gain = [baseline_entropy - feature_entropy(f) for f in range(feature_count)]

    best_feature, best_gain = max(enumerate(information_gain), key = operator.itemgetter(1))

    return best_feature

def potential_leaf_node(data):
    count = calculate_category_frequency(data)
    return count.most_common(1)[0]

def create_tree(data, label):
    category, count = potential_leaf_node(data)
    if count == len(data):
        return category
    node = {}
    feature = best_feature_to_split(data)
    feature_label = label[feature]
    node[feature_label] = {}
    classes = set([d[feature] for d in data])
    for c in classes:
        partitioned = [d for d in data if d[feature] == c]
        node[feature_label][c] = create_tree(partitioned, label)
    return node

def root_node(tree):
    return list(tree.keys())[0]

def is_dictionary(potential_node):
    return isinstance(potential_node, dict)

def classify(tree, label, data):
    root = root_node(tree)
    node = tree[root]
    index = label.index(root)

    for key in node.keys():
        if data[index] == key:
            if is_dictionary(node[key]):
                return classify(node[key], label, data)
            else:
                return node[key]
