import pickle
with open('./data/sensor_graph/adj_pems08.pkl', 'rb') as f:
    adj = pickle.load(f)

print(adj.shape)