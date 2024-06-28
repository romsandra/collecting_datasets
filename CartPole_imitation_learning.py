import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

observation_dim = 4
action_num = 2  # Количество возможных действий: 0 и 1
hidden_units = 256
epochs_num = 10**4
learning_rate = 0.001

class DemoDataset(Dataset):
    """Класс для создания датасета демонстрационных данных"""
    def __init__(self, input_states, input_actions):
        self.states = input_states
        self.actions = input_actions

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

class MLP(nn.Module):
    """Многослойный перцептрон для имитационного обучения"""
    def __init__(self, input_size, action_size, neurons=64):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, neurons)
        self.fc2 = nn.Linear(neurons, neurons)
        self.fc3 = nn.Linear(neurons, action_size)
        self.activation = nn.ReLU()
        self.output_activation = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.output_activation(x)
        return x

def test_agent(agent, env, episodes=5):
    """Функция для тестирования агента в окружении"""
    for episode in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            if agent is None:
                action = env.action_space.sample()
            elif agent is ppo_agent:
                action, _ = agent.predict(obs)
            else:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    action_prob = agent(obs_tensor)
                action = np.argmax(action_prob.numpy())
            result = env.step(action)
            obs, reward, terminated, truncated, info = result
            total_reward += reward
            env.render()
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

if __name__=='__main__':
    # Создание окружения и загрузка модели PPO
    env = gym.make('CartPole-v1')
    ppo_agent = PPO.load('ppo_cartpole', env=env)

    # Сбор демонстрационных данных
    demo_states = []
    demo_actions = []
    for _ in range(5):
        obs, _ = env.reset()
        terminated, truncated = False, False
        while not(terminated or truncated):
            action, _ = ppo_agent.predict(obs)
            res = env.step(action)
            next_obs, reward, terminated, truncated, info = res
            demo_states.append(obs)
            demo_actions.append(action)
            obs = next_obs
    env.close()

    # Преобразование данных в numpy массивы
    demo_states = np.array(demo_states).astype(np.float32)
    demo_actions = np.array(demo_actions).astype(np.int64)

    # Нормализация состояний
    mean = np.mean(demo_states, axis=0)
    std = np.std(demo_states, axis=0)
    demo_states = (demo_states - mean) / (std + 1e-8)

    # Сохранение данных о работе RL-агента в CSV файл
    df = pd.DataFrame({'observations': list(demo_states), 'actions': demo_actions})
    df.to_csv('CartPole_data.csv', index=False)
    print("Демонстрационные данные сохранены в 'CartPole_data.csv' с разделителем ','")

    # Создание датасета и загрузчика данных
    dataset = DemoDataset(demo_states, demo_actions)
    dataloader = DataLoader(dataset, batch_size=hidden_units, shuffle=True)

    # Инициализация модели, оптимизатора и функции потерь
    model = MLP(observation_dim, action_num, hidden_units)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Обучение модели
    for i in range(epochs_num):
        for states, actions in dataloader:
            states = states.float()
            optimizer.zero_grad()
            outputs = model(states)
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()

        # Печать потерь каждые 1000 эпох
        if (i+1) % 1000 == 0:
            print(f'Epoch {i+1}/{epochs_num}, Loss: {loss.item():.4f}')

    # Сохранение обученной модели
    torch.save(model.state_dict(), 'CartPole_im.pth')
    print("Многослойный перцептрон, обученный имитационно, сохранён как 'CartPole_im.pth'")

    # Загрузка обученной имитационной модели
    imitation_model = MLP(observation_dim, action_num, hidden_units)
    imitation_model.load_state_dict(torch.load('CartPole_im.pth'))
    imitation_model.eval()

    # Демонстрация работы модели имитационного обучения и сравнение её работы с работой RL-агента
    print("Testing RL Agent")
    test_agent(ppo_agent, env)

    print("Testing Imitation Model")
    test_agent(imitation_model, env)