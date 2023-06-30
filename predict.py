import os
import torch
import requests
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.data import Dataset
from torch_geometric.nn import GATConv
import csv


############################  class data ############################
class HW3Dataset(Dataset):
    url = 'https://technionmail-my.sharepoint.com/:u:/g/personal/ploznik_campus_technion_ac_il/EUHUDSoVnitIrEA6ALsAK1QBpphP5jX3OmGyZAgnbUFo0A?download=1'

    def __init__(self, root, transform=None, pre_transform=None):
        super(HW3Dataset, self).__init__(root, transform, pre_transform)
    @property
    def raw_file_names(self):
        return ['data.pt']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        file_url = self.url.replace(' ', '%20')
        response = requests.get(file_url)

        if response.status_code != 200:
            raise Exception(f"Failed to download the file, status code: {response.status_code}")

        with open(os.path.join(self.raw_dir, self.raw_file_names[0]), 'wb') as f:
            f.write(response.content)

    def process(self):
        raw_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        data = torch.load(raw_path)
        torch.save(data, self.processed_paths[0])

    @staticmethod
    def len():
        return 1

    def get(self, idx):
        return torch.load(self.processed_paths[0])

############################  class data ############################
class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, num_heads):
        super().__init__()
        self.conv1 = GATConv(dataset.num_features, hidden_channels, heads=num_heads)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads)
        self.classifier = Linear(hidden_channels * num_heads, dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.6, training=self.training)
        out = self.classifier(h)
        return out


if __name__ == '__main__':
    print("-----------creating data set----------------")
    dataset = HW3Dataset(root='data/hw3/')
    data = dataset[0]

    print("-----------loading our best per trained model----------------")
    model = GAT(hidden_channels=100, num_heads=8)
    # Load the state dictionary of the saved model
    state_dict =torch.load("trained_model.pkl")
    # Load the state dictionary into the model
    model.load_state_dict(state_dict)
    # Put the model in evaluation mode
    model.eval()

    print("-----------predicting ----------------")
    # Forward pass through the model to get predictions
    unseen_predictions = model(data.x, data.edge_index)
    unseen_predictions = unseen_predictions.argmax(dim=1)

    print("-----------creating CSV----------------")
    predictions_file = 'predictions.csv'
    with open(predictions_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['idx', 'predictions'])
        for idx, prediction in enumerate(unseen_predictions):
            writer.writerow([idx, prediction.item()])

    print("-----------done!----------------")

