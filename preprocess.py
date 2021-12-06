import collections
import reverse_geocoder as rg
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
	user_to_coordinates = {}
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
			if user_id not in user_to_coordinates:
				user_to_coordinates[user_id] = (lat, lot)

	# write to csv files
	with open(csv_filename, 'w') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=field_names)
		writer.writeheader()
		writer.writerows(data_rows)
		csvfile.flush()
		
	return data_rows, user_to_mentions, user_to_coordinates


def read_dataset_without_saving(filename, csv_filename, data_encoding="ISO-8859-1"):
	user_to_mentions = {}
	user_to_coordinates = {}
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
			if user_id not in user_to_coordinates:
				user_to_coordinates[user_id] = (lat, lot)
		
	return data_rows, user_to_mentions, user_to_coordinates


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
		mention_graph[k] = collections.defaultdict(int)
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


def coordinates_to_cities(user_to_coordinates):
    """
    Convert the coordinates to cities. Return a dict(user_id->city).
    
    Parameters:
        user_to_coordinates: dict(user_id->coordiantes)
    """
    # TODO
    counter = 0
    user_to_city = collections.defaultdict()
    for key, value in user_to_coordinates.items():
        result = rg.search(value)
        user_to_city[key] = result[0]['name']
        print("processing #", counter)
        counter += 1
    return user_to_city


def cities_to_labels(user_to_city):
    """
    Convert city names to integer labels, where each city correspond to 
    one integer. Return a dict(user_id->label) and the number of labels.
    
    Parameters:
        user_to_coordinates: dict(user_id->city)
    """
    city_to_label = {}
    counter = 0
    user_to_label = {}
    for user in user_to_city.keys():
        city = user_to_city[user]
        if not city in city_to_label:
            city_to_label[city] = counter
            counter += 1
        user_to_label[user] = city_to_label[city]
    return user_to_label, counter
        

def build_label_distribution(user_to_label, num_of_labels):
    """
    Build the label distribution of each user.
    
    Parameters:
        user_to_label: dict(user_id->label)
        
    Returns:
        Y: nxc matrix of the original label distribution,
    where n is the number of nodes, c is the number of labels.
    """
    Y = np.zeros((len(user_to_label), num_of_labels))
    for i, (user, label) in enumerate(user_to_label.items()):
        Y[i,label] = 1
    return Y


def build_weight_matrix(mention_graph):
    """
    Build the weight matrix of edges using mention graph.
    
    Parameters:
        mention_graph: a dict(user -> dict(user -> number of bidirectional mentions))
        
    Returns:
        W: nxn weight matrix, where W_ij measures the similarity between
    node i and node j
        
    """
    W = np.zeros((len(mention_graph), len(mention_graph)))
    user_to_index = {}
    for i, (user, neighbor_to_mentions) in enumerate(mention_graph.items()):
        user_to_index[user] = i
    for i, (user, neighbor_to_mentions) in enumerate(mention_graph.items()):
        for neighbor in neighbor_to_mentions.keys():
            W[i, user_to_index[neighbor]] = neighbor_to_mentions[neighbor]
       
    # fill diagonal with the heighest weight + 1
    #np.fill_diagonal(W, np.max(W)+1)
    # fill diagonal with the heighest weight of the row + 1
    #for i in range(W.shape[0]):
    #    W[i,i] = np.max(W[i]+1)
    # make everything less than 1, then fill diagonal with 1
    #W = W/(np.max(W)+1)
    #np.fill_diagonal(W, 1)
    
    return W


def test_build_label_distribution():
    
    user_to_coordinates = {"user_beijing":(39.9042, 116.4074), "user_la":(34.0522, -118.2437), "user2_Beijing": (39.9041, 116.4075)}
    user_to_city = coordinates_to_cities(user_to_coordinates)
    
    print(user_to_city)
    
    user_to_label, counter = cities_to_labels(user_to_city)
    
    print(user_to_label, counter)
    
    Y = build_label_distribution(user_to_label, counter)
    
    print(Y)
    
    
def displace_user_information(idx, idx_to_user, mention_graph, user_to_label, W, Y_pred):
    user = idx_to_user[idx]
    print(user, "'s label: ", user_to_label[user])
    print("data from original graph:")
    print(user, "'s neighbors:")
    for i, (neighbor, num_mentions) in enumerate(mention_graph[user].items()):
        print(neighbor, " ", num_mentions, user_to_label[neighbor])
    print("data from matrix representation:")
    nonzero_indices = np.nonzero(W[idx])[0]
    print(user, "'s neighbors:")
    for neighbor_idx in nonzero_indices:
        print(idx_to_user[neighbor_idx], " ", W[idx, neighbor_idx], np.argmax(Y_pred[neighbor_idx]))


# if __name__ == '__main__':
#data_rows, user_to_mentions = read_dataset(dataset_filename, csv_filename)
data_rows, user_to_mentions, user_to_coordinates = read_dataset(dataset_filename, csv_filename)
print("the first data_rows is: ", data_rows[0])
print(user_to_mentions[data_rows[0]['user_id']])
mention_graph = build_mention_graph(user_to_mentions)
# print(mention_graph)
print("the mention graph is", mention_graph[data_rows[0]['user_id']])
