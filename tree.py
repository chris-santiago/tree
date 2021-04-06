from collections import namedtuple
from operator import attrgetter
from typing import List, Union

import utils


Leaf = namedtuple('Leaf', ['label'])
SplitTest = namedtuple('SplitTest', ['attr', 'value', 'info_gain'])
Node = namedtuple('Node', ['split_attr', 'split_val', 'left', 'right', 'label'])


class DecisionTree:
	def __init__(self, data_attr: List[str], label_attr: str, max_depth=6):
		self.max_depth = max_depth
		self.data_attr = data_attr
		self.label_attr = label_attr
		self.tree = None

	def _make_partitions(self, data: List[namedtuple], split_attr: str, split_val: Union[int, float]) -> Node:
		most_common = utils.most_common_label([getattr(d, self.label_attr) for d in data])
		left = []
		right = []
		for d in data:
			if getattr(d, split_attr) <= split_val:
				left.append(d)
			else:
				right.append(d)
		return Node(split_attr, split_val, left, right, most_common)

	def get_info_gain(self, prior_labels: List[str], current_part: Node) -> float:
		entropy_prior = utils.get_entropy(prior_labels)

		labels_left = [getattr(d, self.label_attr) for d in current_part.left]
		entropy_left = utils.get_entropy(labels_left)
		prob_left = len(labels_left) / len(prior_labels)

		labels_right = [getattr(d, self.label_attr) for d in current_part.right]
		entropy_right = utils.get_entropy(labels_right)
		prob_right = len(labels_right) / len(prior_labels)

		return entropy_prior - ((entropy_left * prob_left) + (entropy_right * prob_right))

	def get_best_split(self, data: List[namedtuple], features: List[str]):
		labels = [getattr(d, self.label_attr) for d in data]
		results = []
		for f in features:
			for val in [getattr(d, f) for d in data]:
				part = self._make_partitions(data, f, val)
				gain = self.get_info_gain(labels, part)
				results.append(SplitTest(f, val, gain))
		best = max(results, key=attrgetter('info_gain'))
		return self._make_partitions(data, best.attr, best.value)

	def fit(self, data: List[namedtuple], depth: int = 0) -> Node:
		label_counts = utils.get_label_counts(data, self.label_attr)
		most_common = utils.most_common_label([getattr(d, self.label_attr) for d in data])
		features = self.data_attr

		if len(label_counts) == 1:
			return Leaf(most_common)

		elif not features:
			return Leaf(most_common)

		elif depth > self.max_depth:
			return Leaf(most_common)

		else:
			node = self.get_best_split(data, features)
			features.remove(node.split_attr)
			left = self.fit(node.left, depth+1)
			right = self.fit(node.right, depth+1)
			depth += 1
			self.tree = Node(node.split_attr, node.split_val, left, right, node.label)
			return self.tree

	def predict(self, data: List[namedtuple]):
		node = self.tree
		while isinstance(node, Node):
			if getattr(data, node.split_attr) <= node.split_val:
				node = node.left
			else:
				node = node.right
		return node.label


if __name__ == '__main__':
	data = utils.csv_to_tuples('iris.txt')
	features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
	tree = DecisionTree(features, 'species')
	tree.fit(data)
	print('Fitted tree:')
	print(tree.tree)

	TestData = namedtuple('TestData', ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
	testSetosa = TestData(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2)
	testVersicolor = TestData(sepal_length=6.4, sepal_width=2.9, petal_length=4.3, petal_width=1.3)
	pred = tree.predict(testVersicolor)
	print('-----')
	print(f'Prediction: {pred}')
