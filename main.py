import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Global constants, TODO: edit based on model architecture/dataset/task
metric_space_dim = 2
embedding_dim = 2
input_dim = 2
hidden_dim = 4
K = 3

# Learn transformation f, prototypes P, and user weights w_i
class PreferenceNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_users, num_prototypes):
        super(PreferenceNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=False)
        )
        
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, embedding_dim))
        self.user_weights = nn.Parameter(torch.rand(num_users, num_prototypes))
        
    def get_user_ideal_point(self, user_id):
        w_i = F.softmax(self.user_weights[user_id], dim=0)
        return torch.matmul(w_i, self.prototypes)
    
    def forward(self, x):
        return self.network(x)

# Dataset class wiht preferences pairs + user IDs (@emily each part of the preference pair is like f(response, prompt))
class PreferenceDataset(Dataset):
    def __init__(self, features, preferences, user_ids):
        self.features = features
        self.preferences = preferences
        self.user_ids = user_ids
        
    def __len__(self):
        return len(self.preferences)
    
    def __getitem__(self, idx):
        i, j = self.preferences[idx]
        user_id = self.user_ids[idx]
        return self.features[i], self.features[j], user_id

# Synthetic dataset to reproduce exp from section 4.1
def generate_synthetic_data_with_prototypes(n_samples, feature_dim, n_pairs_per_user, n_users, K):
    prototypes = np.random.randn(K, feature_dim)
    user_weights = np.random.dirichlet(np.ones(K), size=n_users)
    user_ideal_points = np.dot(user_weights, prototypes)
    
    features = np.random.randn(n_samples, feature_dim)
    
    secret_transform = np.random.randn(feature_dim, feature_dim)
    
    transformed_features = np.dot(features, secret_transform)
    transformed_prototypes = np.dot(prototypes, secret_transform)
    transformed_ideal_points = np.dot(user_ideal_points, secret_transform)
    
    all_preferences = []
    all_user_ids = []
    
    for user_id in range(n_users):
        user_ideal = transformed_ideal_points[user_id]
        
        for _ in range(n_pairs_per_user):
            # Select two random points
            idx1, idx2 = np.random.choice(n_samples, size=2, replace=False)
            dist1 = np.linalg.norm(transformed_features[idx1] - user_ideal)
            dist2 = np.linalg.norm(transformed_features[idx2] - user_ideal)
            
            if dist1 < dist2:
                all_preferences.append([idx1, idx2])
            else:
                all_preferences.append([idx2, idx1])
            
            all_user_ids.append(user_id)
    
    return (torch.FloatTensor(features), 
            np.array(all_preferences), 
            np.array(all_user_ids), 
            prototypes,
            user_ideal_points)

# Plotting helper
def plot_prototype_analysis(model, true_prototypes, true_ideal_points):
    if K != 3 or metric_space_dim != 2:
        print("Plotting only supported for K=3 and 2 dimensions")
        return
        
    predicted_ideal_points = np.array([model.get_user_ideal_point(i).detach().numpy() 
                                     for i in range(len(true_ideal_points))])
    
    predicted_prototypes = model.prototypes.detach().numpy()
    
    plt.figure(figsize=(10, 10))
    
    plt.plot(true_prototypes[[0,1,2,0], 0], true_prototypes[[0,1,2,0], 1], 
             'k-', label='True Prototype Triangle')
    plt.scatter(true_prototypes[:,0], true_prototypes[:,1], 
               color='red', s=100, label='True Prototypes')
    
    plt.plot(predicted_prototypes[[0,1,2,0], 0], predicted_prototypes[[0,1,2,0], 1],
             'k--', label='Predicted Prototype Triangle')
    plt.scatter(predicted_prototypes[:,0], predicted_prototypes[:,1],
               color='orange', s=100, label='Predicted Prototypes')
    
    plt.scatter(true_ideal_points[:,0], true_ideal_points[:,1], 
               color='blue', alpha=0.5, label='True Ideal Points')
    plt.scatter(predicted_ideal_points[:,0], predicted_ideal_points[:,1], 
               color='green', alpha=0.5, label='Predicted Ideal Points')
    
    plt.legend()
    plt.title('Prototype Analysis')
    plt.grid(True)
    plt.savefig('plot.png')

def train_preference_model(model, train_loader, optimizer, n_epochs=100):
    criterion = nn.MarginRankingLoss(margin=1.0)
    
    for epoch in range(n_epochs):
        total_loss = 0
        for preferred, non_preferred, user_ids in train_loader:
            # Forward pass through network
            preferred_embedding = model(preferred)
            non_preferred_embedding = model(non_preferred)
            
            # Get ideal points for each user in batch
            ideal_points = torch.stack([model.get_user_ideal_point(uid) for uid in user_ids])
            
            preferred_dist = torch.norm(preferred_embedding - ideal_points, dim=1)
            non_preferred_dist = torch.norm(non_preferred_embedding - ideal_points, dim=1)
            target = torch.ones(preferred.size(0))

            loss = criterion(non_preferred_dist, preferred_dist, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {total_loss/len(train_loader):.4f}')

if __name__ == "__main__":
    # Generate synthetic data
    features, preferences, user_ids, true_prototypes, true_ideal_points = generate_synthetic_data_with_prototypes(
        n_samples=1000, feature_dim=metric_space_dim, n_pairs_per_user=1000, n_users=50*K, K=K)
    
    dataset = PreferenceDataset(features, preferences, user_ids)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Init model
    n_users = len(np.unique(user_ids))
    model = PreferenceNet(input_dim=input_dim, hidden_dim=hidden_dim, 
                         output_dim=metric_space_dim, num_users=n_users,
                         num_prototypes=K)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_preference_model(model, train_loader, optimizer)
    plot_prototype_analysis(model, true_prototypes, true_ideal_points)
