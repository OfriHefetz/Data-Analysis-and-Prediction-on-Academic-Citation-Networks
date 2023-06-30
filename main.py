import requests
import os
from torch_geometric.data import Dataset
import torch
import numpy as np
import networkx as nx
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GATConv
import pickle



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

    def len(self):
        return 1

    def get(self, idx):
        return torch.load(self.processed_paths[0])

######### option 1 - GCN with 3 conv layers  ##########
# class GCN(torch.nn.Module):
#     def __init__(self, hidden_channels, dropout, activation):
#         super().__init__()
#         torch.manual_seed(1234)
#         self.conv1 = GCNConv(dataset.num_features, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, hidden_channels)
#         self.conv3 = GCNConv(hidden_channels, hidden_channels // 2)
#         self.classifier = Linear(hidden_channels // 2, dataset.num_classes)
#         self.dropout = dropout
#         self.activation = activation
#
#     def forward(self, x, edge_index):
#         h = self.conv1(x, edge_index)
#         h = self.activation(h)
#         h = F.dropout(h, p=self.dropout, training=self.training)
#         h = self.conv2(h, edge_index)
#         h = self.activation(h)
#         h = F.dropout(h, p=self.dropout, training=self.training)
#         h = self.conv3(h, edge_index)
#         h = self.activation(h)
#         h = F.dropout(h, p=self.dropout, training=self.training)
#         out = self.classifier(h)
#         return out


######### option 2 - GCN with 2 conv layers  ##########
# class GCN(torch.nn.Module):
#     def __init__(self, hidden_channels, dropout, activation):
#         super().__init__()
#         torch.manual_seed(1234)
#         self.conv1 = GCNConv(dataset.num_features, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, hidden_channels // 2)
#         self.classifier = Linear(hidden_channels // 2, dataset.num_classes)
#         self.dropout = dropout
#         self.activation = activation
#
#     def forward(self, x, edge_index):
#         h = self.conv1(x, edge_index)
#         h = self.activation(h)
#         h = F.dropout(h, p=self.dropout, training=self.training)
#         h = self.conv2(h, edge_index)
#         out = self.classifier(h)
#         return out


######### option 3 - SAGE with 2 conv layers  ##########
# class GraphSAGE(torch.nn.Module):
#     def __init__(self, hidden_channels, dropout, activation):
#         super().__init__()
#         self.conv1 = SAGEConv(dataset.num_features, hidden_channels)
#         self.conv2 = SAGEConv(hidden_channels, hidden_channels)
#         self.classifier = Linear(hidden_channels, dataset.num_classes)
#         self.dropout = dropout
#         self.activation = activation
#
#     def forward(self, x, edge_index):
#         h = self.conv1(x, edge_index)
#         h = self.activation(h)
#         h = F.dropout(h, p=0.6, training=self.training)
#         h = self.conv2(h, edge_index)
#         h = self.activation(h)
#         h = F.dropout(h, p=0.6, training=self.training)
#         out = self.classifier(h)
#         return out


# class GAT(torch.nn.Module):
#     def __init__(self, hidden_channels, num_heads):
#         super().__init__()
#         self.conv1 = GATConv(dataset.num_features, hidden_channels, heads=num_heads)
#         self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads)
#         self.classifier = Linear(hidden_channels * num_heads, dataset.num_classes)
#
#     def forward(self, x, edge_index):
#         h = self.conv1(x, edge_index)
#         h = F.relu(h)
#         h = F.dropout(h, p=0.5, training=self.training)
#         h = self.conv2(h, edge_index)
#         h = F.relu(h)
#         h = F.dropout(h, p=0.5, training=self.training)
#         out = self.classifier(h)
#         return out
#
#
# def train():
#     model.train()
#     optimizer.zero_grad()
#     out = model(data.x, data.edge_index)
#     a = data.y[data.train_mask].resize_(len(data.train_mask))
#     loss = criterion(out[data.train_mask], a)
#     loss.backward()
#     optimizer.step()
#     return loss
#
#
# def test():
#     model.eval()
#     out = model(data.x, data.edge_index)
#     pred = out.argmax(dim=1)
#     b = data.y[data.val_mask].resize_(len(data.val_mask))
#     test_correct = pred[data.val_mask] == b
#     test_acc = int(test_correct.sum()) / len(data.val_mask)
#     return test_acc



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
        h = F.dropout(h, p=0.5, training=self.training)
        out = self.classifier(h)
        return out

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    a = data.y[data.train_mask].squeeze()
    loss = criterion(out[data.train_mask], a)
    loss.backward()
    optimizer.step()
    return loss


def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    b = data.y[data.val_mask].resize_(len(data.val_mask))
    test_correct = pred[data.val_mask] == b
    test_acc = int(test_correct.sum()) / len(data.val_mask)
    return test_acc


if __name__ == '__main__':
    np.random.seed(2023)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("------------data set----------------")
    dataset = HW3Dataset(root='data/hw3/')
    data = dataset[0]


    print("------------defining model----------------")
    ## option 1
    # model = GCN(hidden_channels=180, dropout=0.6, activation=torch.relu)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    # criterion = torch.nn.CrossEntropyLoss()

    ## option 2
    # model = GraphSAGE(hidden_channels=180)  # Adjust the hidden_channels value as desired
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.004, weight_decay=5e-4)
    # criterion = torch.nn.CrossEntropyLoss()

    ## option 3
    model = GAT(hidden_channels=100, num_heads=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    print("------------starting stage train----------------")
    total_loss = []
    for i, epoch in enumerate(range(1, 300)):
        loss = train()
        total_loss.append(loss)
        if i % 10 == 0:
            print(f'Epoch: {epoch}, Loss: {loss:.4f}')

    print("------------starting stage eval----------------")
    val_acc = test()
    print(f'Validation Accuracy: {val_acc:.4f}')
    print(f'Minimum Loss Value: {loss:.4f}')

    print("------------saving the model----------------")
    torch.save(model.state_dict(), '/home/student/HW3/trained_model.pkl')


