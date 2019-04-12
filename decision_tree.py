import pickle
import collections
import math
import operator

with open("data_rand", "rb") as f:
    training_data = pickle.load(f)
    print(training_data)

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

def root(tree):
    return list(tree.keys())[0]

def is_dictionary(potential_node):
    return isinstance(potential_node, dict)

def classify(tree, label, data):
    root_node = root(tree)
    child_node = tree[root_node]
    index = label.index(root_node)

    for key in child_node.keys():
        if data[index] == key:
            if is_dictionary(child_node[key]):
                return classify(child_node[key], label, data)
            else:
                return child_node[key]

def build_rule(tree, label, identifier=0):
    space_identifier = '  '*identifier
    space_identifier_copy = space_identifier
    root_node = root(tree)
    child_node = tree[root_node]
    index = label.index(root_node)

    for key in child_node.keys():
        space_identifier_copy += 'if ' + label[index] + ' = ' + str(key)
        if is_dictionary(child_node[key]):
            space_identifier_copy += ':\n' + space_identifier + build_rule(child_node[key], label, identifier + 1)
        else:
            space_identifier_copy += ' then ' + str(child_node[key]) + ('.\n' if identifier == 0 else ', ')

    if space_identifier_copy[-2:] == ', ':
        space_identifier_copy = space_identifier_copy[:-2]

    space_identifier_copy += '\n'
    return space_identifier_copy

def find_edges(tree, label, x, y):
    x.sort()
    y.sort()
    diagonals = [i for i in set(x).intersection(set(y))]
    diagonals.sort()
    L = [classify(tree, label, [d, d]) for d in diagonals]
    low = L.index(False)
    min_x = x[low]
    min_y = y[low]

    high = L[::-1].index(False)
    max_x = x[len(x)-1-high]
    max_y = y[len(y)-1-high]

    return (min_x, min_y), (max_x, max_y)

label = ['x', 'y', 'out']

tree = create_tree(training_data, label)
print(build_rule(tree, label))

print(find_edges(tree, label, [x[0] for x in training_data], [x[0] for x in training_data]))