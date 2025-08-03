import torch
import torch.nn as nn
import torch.optim as optim


class PolicyGradient:
    def __init__(self, state_dim, action_dim, lr=0.01):
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1),
        )
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.saved_log_probs = []
        self.rewards = []

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.network(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        self.saved_log_probs.append(action_dist.log_prob(action))
        return action.item()

    def update(self):
        R = 0
        policy_loss = []
        returns = []

        for r in self.rewards[::-1]:
            R = r + 0.99 * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Calculate loss
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        del self.rewards[:]
        del self.saved_log_probs[:]
