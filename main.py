import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datasets import load_dataset
from scipy.spatial import ConvexHull
import gc
import pickle


# Global constants, TODO: edit based on model architecture/dataset/task

#synthetic
# metric_space_dim = 2
# embedding_dim = 2
# input_dim = 2
# hidden_dim = 4
# K = 3

# gemma
metric_space_dim = 16
embedding_dim = 2304
input_dim = 2*embedding_dim
hidden_dim = 1024

model_name = "google/gemma-2-2b"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)


def get_embedding_vector_for_string(prompt, model, tokenizer):
    # Tokenize the input string and move to the model's device
    seq_ids = tokenizer(prompt, return_tensors='pt', truncation=True, padding=True).to(model.device)["input_ids"]

    # Compute the embeddings
    with torch.no_grad():
        output = model(seq_ids)
        embedding = output["last_hidden_state"].mean(dim=1).squeeze()

    del seq_ids, output
    torch.cuda.empty_cache()
    gc.collect()

    return embedding  # Return detached embedding

def extract_openai_data(example):
    prompt = example["info"]["post"]
    summaries = example["summaries"]
    chosen_idx = example["choice"]

    prompt = get_embedding_vector_for_string(prompt, model, tokenizer)
    positive = get_embedding_vector_for_string(summaries[chosen_idx]["text"], model, tokenizer)
    negative = get_embedding_vector_for_string(summaries[1 - chosen_idx]["text"], model, tokenizer)

    user_id = example["worker"]

    return {"prompt": prompt, "positive": positive, "negative": negative, "user_id": user_id}

def extract_chatbotarena_data(example):
    winner = "conversation_a" if example["winner"] == "model_a" else "conversation_b"
    loser = "conversation_b" if winner == "model_a" else "conversation_a"

    prompt = get_embedding_vector_for_string(example[winner][0]["content"], model, tokenizer)
    positive = get_embedding_vector_for_string(example[winner][1]["content"], model, tokenizer)
    negative = get_embedding_vector_for_string(example[loser][1]["content"], model, tokenizer)
    user_id = example["judge"]

    return {"prompt": prompt, "positive": positive, "negative": negative, "user_id": user_id}

def retrieve_info_from_data(data):
    positives = torch.stack([torch.cat((item["positive"].cpu(), item["prompt"].cpu()), dim=0) for item in data])
    negatives = torch.stack([torch.cat((item["negative"].cpu(), item["prompt"].cpu()), dim=0) for item in data])
    user_ids = [item["user_id"] for item in data]

    features = torch.cat((positives, negatives), dim=0)
    preferences = torch.tensor([(i, i + len(data)) for i in range(len(data))], dtype=torch.int64)

    return features, preferences, user_ids

# Learn transformation f, prototypes P, and user weights w_i
class PreferenceNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_users, num_prototypes):
        super(PreferenceNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, output_dim)
        )

        self.prototypes = nn.Parameter(torch.randn(num_prototypes, input_dim))
        self.user_weights = nn.Parameter(torch.rand(num_users, num_prototypes))

    def get_user_ideal_point(self, user_id):
        w_i = F.softmax(self.user_weights[user_id], dim=0)
        return torch.matmul(w_i, self.prototypes)

    def forward(self, x):
        return self.network(x)



# Dataset class wiht preferences pairs + user IDs (@emily each part of the preference pair is like f(response, prompt))
class PreferenceDataset(Dataset):
    def __init__(self, features, preferences, user_ids):
        # Canonicalize user IDs to integers
        unique_users = sorted(set(user_ids))
        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.idx_to_user_id = {idx: uid for uid, idx in self.user_id_to_idx.items()}

        # Convert user IDs to integer indices
        canonical_user_ids = torch.tensor([self.user_id_to_idx[uid] for uid in user_ids])
        n_users = len(unique_users)

        # Create indices for each user's preferences
        user_indices = {i: [] for i in range(n_users)}
        for idx, uid in enumerate(canonical_user_ids):
            user_indices[uid.item()].append(idx)

        # Split indices for each user
        train_idx, val_idx, test_idx = [], [], []
        for uid, indices in user_indices.items():
            n = len(indices)
            # Ensure at least 1 sample per user in each split
            n_train = min(int(0.8 * n), n - 2)
            n_val = min(max(1, int(0.1 * n)),n-n_train-1)
            n_test = n - n_train - n_val

            # Randomly shuffle indices
            shuffled = torch.randperm(len(indices))
            user_samples = [indices[i] for i in shuffled]

            train_idx.extend(user_samples[:n_train])
            val_idx.extend(user_samples[n_train:n_train+n_val])
            test_idx.extend(user_samples[n_train+n_val:])

        # Store split data
        self.train_preferences = preferences[train_idx]
        self.train_user_ids = canonical_user_ids[train_idx]
        self.val_preferences = preferences[val_idx]
        self.val_user_ids = canonical_user_ids[val_idx]
        self.test_preferences = preferences[test_idx]
        self.test_user_ids = canonical_user_ids[test_idx]

        # Store features
        self.train_features = features
        self.val_features = features
        self.test_features = features

        # Default to training set
        self.features = self.train_features
        self.preferences = self.train_preferences
        self.user_ids = self.train_user_ids

    def set_split(self, split='train'):
        if split == 'train':
            self.features = self.train_features
            self.preferences = self.train_preferences
            self.user_ids = self.train_user_ids
        elif split == 'val':
            self.features = self.val_features
            self.preferences = self.val_preferences
            self.user_ids = self.val_user_ids
        else:  # test
            self.features = self.test_features
            self.preferences = self.test_preferences
            self.user_ids = self.test_user_ids

    def get_original_user_id(self, idx):
        return self.idx_to_user_id[idx]

    def __len__(self):
        return len(self.preferences)

    def __getitem__(self, idx):
        i, j = self.preferences[idx]
        user_id = self.user_ids[idx]
        return self.features[i], self.features[j], user_id

# Synthetic dataset to reproduce exp from section 4.1
def generate_synthetic_data_with_prototypes(n_samples, feature_dim, n_pairs_per_user, n_users, K):
    # Sample prototypes with minimum distance constraint
    min_distance = 0.2  # Minimum distance between any two prototypes
    prototypes = np.zeros((K, feature_dim))
    for i in range(K):
        while True:
            # Sample a candidate prototype
            candidate = (1/np.sqrt(feature_dim)) * np.random.randn(feature_dim)
            if i == 0:  # First prototype can be accepted immediately
                prototypes[0] = candidate
                break
            # Check distances to existing prototypes
            distances = np.linalg.norm(prototypes[:i] - candidate, axis=1)
            if np.all(distances[distances > 0] >= min_distance):
                prototypes[i] = candidate
                break
    user_weights = np.random.dirichlet(np.ones(K), size=n_users)
    user_ideal_points = np.dot(user_weights, prototypes)

    features = (1/np.sqrt(feature_dim)) * np.random.randn(n_samples, feature_dim)

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
def plot_prototype_analysis(model, true_prototypes, true_ideal_points, filename='plot.png'):
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
    plt.savefig(filename)
    plt.close()

def plot_prototype_analysis_experimental(model, true_prototypes, true_ideal_points, filename='plot.png'):
    if metric_space_dim != 2:
        print("Plotting only supported for 2 dimensions")
        return

    little_k = len(true_prototypes)  # Get number of prototypes from tensor
    predicted_ideal_points = np.array([model.get_user_ideal_point(i).detach().numpy()
                                     for i in range(len(true_ideal_points))])

    predicted_prototypes = model.prototypes.detach().numpy()

    plt.figure(figsize=(10, 10))

    # Get convex hull indices for true prototypes
    hull = ConvexHull(true_prototypes)
    hull_indices = np.append(hull.vertices, hull.vertices[0])  # Close the polygon

    # Plot true prototypes convex hull and vertices
    plt.plot(true_prototypes[hull_indices, 0], true_prototypes[hull_indices, 1],
             'k-', label='True Prototype Hull')
    for i in range(len(hull.vertices)):
        j = (i + 1) % len(hull.vertices)
        plt.plot([true_prototypes[hull.vertices[i], 0], true_prototypes[hull.vertices[j], 0]],
                [true_prototypes[hull.vertices[i], 1], true_prototypes[hull.vertices[j], 1]], 'k-')
    plt.scatter(true_prototypes[:,0], true_prototypes[:,1],
               color='red', s=100, label='True Prototypes')

    # Get convex hull indices for predicted prototypes
    hull_pred = ConvexHull(predicted_prototypes)
    hull_pred_indices = np.append(hull_pred.vertices, hull_pred.vertices[0])

    # Plot predicted prototypes convex hull and vertices
    plt.plot(predicted_prototypes[hull_pred_indices, 0], predicted_prototypes[hull_pred_indices, 1],
             'k--', label='Predicted Prototype Hull')
    for i in range(len(hull_pred.vertices)):
        j = (i + 1) % len(hull_pred.vertices)
        plt.plot([predicted_prototypes[hull_pred.vertices[i], 0], predicted_prototypes[hull_pred.vertices[j], 0]],
                [predicted_prototypes[hull_pred.vertices[i], 1], predicted_prototypes[hull_pred.vertices[j], 1]], 'k--')
    plt.scatter(predicted_prototypes[:,0], predicted_prototypes[:,1],
               color='orange', s=100, label='Predicted Prototypes')

    plt.scatter(true_ideal_points[:,0], true_ideal_points[:,1],
               color='blue', alpha=0.5, label='True Ideal Points')
    plt.scatter(predicted_ideal_points[:,0], predicted_ideal_points[:,1],
               color='green', alpha=0.5, label='Predicted Ideal Points')

    plt.legend()
    plt.title(f'Prototype Analysis (K={little_k})')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def train_preference_model(model, train_loader, val_loader, optimizer, n_epochs=100):
    criterion = nn.MarginRankingLoss(margin=1.0)
    best_val_acc = 0
    best_model_state = None

    for epoch in range(n_epochs):
        # Training
        model.train()
        total_loss = 0
        for preferred, non_preferred, user_ids in train_loader:
            # Forward pass through network
            preferred_embedding = model(preferred)
            non_preferred_embedding = model(non_preferred)

            # Get ideal points for each user in batch
            ideal_points = torch.stack([model(model.get_user_ideal_point(uid)) for uid in user_ids])

            preferred_dist = torch.norm(preferred_embedding - ideal_points, dim=1)
            non_preferred_dist = torch.norm(non_preferred_embedding - ideal_points, dim=1)
            target = torch.ones(preferred.size(0))

            loss = criterion(non_preferred_dist, preferred_dist, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for preferred, non_preferred, user_ids in val_loader:
                preferred_embedding = model(preferred)
                non_preferred_embedding = model(non_preferred)
                ideal_points = torch.stack([model(model.get_user_ideal_point(uid)) for uid in user_ids])

                preferred_dist = torch.norm(preferred_embedding - ideal_points, dim=1)
                non_preferred_dist = torch.norm(non_preferred_embedding - ideal_points, dim=1)

                correct += (preferred_dist < non_preferred_dist).sum().item()
                total += preferred.size(0)

        val_acc = 100 * correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {total_loss/len(train_loader):.4f}, Val Acc: {val_acc:.2f}%')

    return best_val_acc, best_model_state

if __name__ == "__main__":
    # Generate synthetic data
    # features, preferences, user_ids, true_prototypes, true_ideal_points = generate_synthetic_data_with_prototypes(n_samples=1000, feature_dim=metric_space_dim, n_pairs_per_user=1000, n_users=50*K, K=K)

    # # Use OpenAI dataset
    # dataset = load_dataset("openai/summarize_from_feedback", "comparisons")
    # extracted_data = dataset.map(extract_openai_data)

    # # Use Chatbot Arena dataset
    # dataset = load_dataset("chatbot_arena_conversation")
    # extracted_data = dataset.map(extract_chatbotarena_data)

    with open('chatbot_extracted_data.pkl', 'rb') as f:
        extracted_data = pickle.load(f)


    features, preferences, user_ids = retrieve_info_from_data(extracted_data)
    dataset = PreferenceDataset(features, preferences, user_ids)
    train_dataset = PreferenceDataset(features, preferences, user_ids)
    train_dataset.set_split('train')
    val_dataset = PreferenceDataset(features, preferences, user_ids)
    val_dataset.set_split('val')
    test_dataset = PreferenceDataset(features, preferences, user_ids)
    test_dataset.set_split('test')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # Try different numbers of prototypes
    prototype_results = {}
    best_model_states = {}
    test_results = {}
    for num_prototypes in range(2, 6):
        print(f"\nTraining with {num_prototypes} prototypes:")


        # Init model
        n_users = len(dataset.user_id_to_idx)
        model = PreferenceNet(input_dim=input_dim, hidden_dim=hidden_dim,
                            output_dim=metric_space_dim, num_users=n_users,
                            num_prototypes=num_prototypes)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        val_acc, model_state = train_preference_model(model, train_loader, val_loader, optimizer)
        prototype_results[num_prototypes] = val_acc
        best_model_states[num_prototypes] = model_state

        # Evaluate on test set
        model.load_state_dict(model_state)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for preferred, non_preferred, user_ids in test_loader:
                preferred_embedding = model(preferred)
                non_preferred_embedding = model(non_preferred)
                ideal_points = torch.stack([model(model.get_user_ideal_point(uid)) for uid in user_ids])

                preferred_dist = torch.norm(preferred_embedding - ideal_points, dim=1)
                non_preferred_dist = torch.norm(non_preferred_embedding - ideal_points, dim=1)

                correct += (preferred_dist < non_preferred_dist).sum().item()
                total += preferred.size(0)

        test_acc = 100 * correct / total
        test_results[num_prototypes] = test_acc

        # Plot for synthewtic data case
        if num_prototypes >= 3 and metric_space_dim == 2:
            plot_prototype_analysis_experimental(model, true_prototypes, true_ideal_points, f'prototypes_{num_prototypes}.png')

    print("\nResults for different numbers of prototypes:")
    for num_prototypes in range(2, 6):
        print(f"Number of prototypes: {num_prototypes}")
        print(f"  Validation accuracy: {prototype_results[num_prototypes]:.2f}%")
        print(f"  Test accuracy: {test_results[num_prototypes]:.2f}%")

    # Find best number of prototypes based on validation accuracy
    best_num_prototypes = max(prototype_results.items(), key=lambda x: x[1])[0]
    print(f"\nBest model based on validation accuracy has {best_num_prototypes} prototypes")
    print(f"Its test accuracy: {test_results[best_num_prototypes]:.2f}%")

    # Plot test accuracies
    plt.figure(figsize=(10, 6))
    prototypes = list(range(2, 6))
    accuracies = [test_results[k] for k in prototypes]
    plt.bar(prototypes, accuracies)
    plt.xlabel('Number of Prototypes (K)')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy vs Number of Prototypes')
    plt.grid(True, alpha=0.3)
    for i, v in enumerate(accuracies):
        plt.text(i + 2, v + 1, f'{v:.1f}%', ha='center')

    plt.savefig('best_K.png')
    plt.close()
