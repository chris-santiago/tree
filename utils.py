from typing import List, Union
from collections import Counter, namedtuple
import csv
from math import log


def parse_numbers(row: List[str]):
	parsed = []
	for r in row:
		try:
			parsed.append(float(r))
		except ValueError:
			parsed.append(r)
	return parsed


def csv_to_tuples(file: str) -> List[namedtuple]:
	with open(file, 'r') as fp:
		reader = csv.reader(fp, delimiter=',')
		rows = [x for x in reader]
		attributes = rows[0]
		Data = namedtuple('Data', attributes)
		return [Data._make(parse_numbers(row)) for row in rows[1:]]


def get_probabilities(labels: List) -> List[float]:
	return [count / len(labels) for count in Counter(labels).values()]


def get_entropy(labels: List) -> float:
	return sum(-p * log(p, 2) for p in get_probabilities(labels))


def get_label_counts(data: List[namedtuple], label_attr: str) -> Counter:
	return Counter(getattr(x, label_attr) for x in data)


def most_common_label(labels: List) -> str:
	return Counter(labels).most_common(1)[0][0]


if __name__ == '__main__':
	data = csv_to_tuples('iris.txt')
	print(len(get_label_counts(data, 'species')))
	labels = [x.species for x in data]
	print(get_probabilities(labels))
	print(get_entropy(labels))
