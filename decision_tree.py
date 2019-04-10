import pickle
import collections
import math

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
