from collections import defaultdict
from typing import DefaultDict
import numpy as np
import re,csv

dataset_filename = "GeoText.2010-10-12/full_text.txt"
csv_filename = "data.csv"
data_encoding= "ISO-8859-1"


def clean_line(line):
	line = re.sub(r'[^\x00-\x7F]+',' ',line.strip())
	line = re.sub(r'[^\w\s@,.:!;]',' ',line)
	return line
	

def extract_mentions(line):
	mentions = []
	words = line.strip().split()
	for w in words:
		if w.startswith('@USER_'):
			w = w[1:]
			w = w.rstrip(':')
			mentions.append(w)
	return mentions

def read_dataset(filename, csv_filename, data_encoding="ISO-8859-1"):
	user_to_mentions = {}
	data_rows = []
	field_names = ["user_id", "timestamp", "latitude", "longitude", "raw_text", "mentions"]
	# read raw text files
	with open(filename, 'r', encoding=data_encoding) as f:
		lines = f.readlines()
		#lines = [line.encode('utf-8', errors='ignore').decode('utf-8') for line in lines]
		#print(lines[0])
		for line in lines:
			fields = line.split('\t')
			if len(fields) != 6:
				raise RuntimeError("Num of fields do not match: ".format(fields))
			row = {}
			user_id, ts, lat, lot = fields[0], fields[1], fields[3], fields[4]
			raw_text = clean_line(fields[5])
			mentions = extract_mentions(fields[5])
			row['user_id'] = user_id
			row['timestamp'] = ts
			row['latitude'] = lat
			row['longitude'] = lot
			row['raw_text'] = raw_text
			row['mentions'] = mentions
			if user_id not in user_to_mentions:
				user_to_mentions[user_id] = []
			user_to_mentions[user_id] += mentions
			data_rows.append(row)

	# write to csv files
	with open(csv_filename, 'w') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=field_names)
		writer.writeheader()
		writer.writerows(data_rows)
		csvfile.flush()
		
	return data_rows, user_to_mentions

'''
def build_mention_graph(user_to_mentions):
	mention_graph = {}
	for k, v in user_to_mentions.items():
		mention_graph[k] = []
		potential_neighbors = set(v)
		for potential_neighbor in potential_neighbors:
			if potential_neighbor in user_to_mentions.keys() and k in user_to_mentions[potential_neighbor]:
				mention_graph[k].append((potential_neighbor, v.count(potential_neighbor) + user_to_mentions[potential_neighbor].count(k)))

	return mention_graph
'''

# build bidirectional mention_graph from mention records
# return:
#     mention_graph: a dict(user -> dict(user -> number of bidirectional mentions))
def build_mention_graph(user_to_mentions):
	n_users = len(user_to_mentions)
	dangle_users = set()
	n_edges, n_mentions = 0, 0
	mention_graph = {}
	for k in user_to_mentions.keys():
		# initialize adjacency list
		mention_graph[k] = defaultdict(int)
	print("Start building mention graph...")
	for k, mentions in user_to_mentions.items():
		neighbors,counts = np.unique(mentions, return_counts=True)
		for v,count in zip(neighbors, counts):
			if v in mention_graph.keys():
				# if u mentions v and v also mentions u
				mention_graph[k][v]+=count
				mention_graph[v][k]+=count
				n_edges+=1
				n_mentions+=count
			else:
				dangle_users.add(v)
				#print("Dangling user: {}".format(v))
				#v do not exist in the dataset
	print("Finish building mention graph.")
	print("    nodes: {} edges: {} mentions: {} dangles: {}".format(n_users, n_edges, n_mentions, len(dangle_users)))
	return mention_graph

		



if __name__ == '__main__':
	data_rows, user_to_mentions = read_dataset(dataset_filename, csv_filename)
	print(data_rows[0])
	print(user_to_mentions[data_rows[0]['user_id']])
	mention_graph = build_mention_graph(user_to_mentions)
	print(mention_graph[data_rows[0]['user_id']])

  