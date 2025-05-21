import numpy as np
import matplotlib.pyplot as plt

class WeeklyPattern:
    def __init__(self, points_per_week=168, normalize_by_week=False):
        self.points_per_week = points_per_week
        self.normalize_by_week = normalize_by_week
        self.mean_pattern = None  # [points_per_week, D]
        self.std_pattern = None   # [points_per_week, D]

    def fit(self, data):  # data: [N, D] np.ndarray
        N, D = data.shape
        num_weeks = N // self.points_per_week
        data = data[:num_weeks * self.points_per_week]  # truncate to full weeks
        weekly = data.reshape(num_weeks, self.points_per_week, D)  # [W, points_per_week, D]

        if self.normalize_by_week:
            # 按周归一化
            means = weekly.mean(axis=1, keepdims=True)  # [W, 1, D]
            stds = weekly.std(axis=1, keepdims=True) + 1e-6  # [W, 1, D]
            norm_weekly = (weekly - means) / stds
            self.mean_pattern = norm_weekly.mean(axis=0)  # [points_per_week, D]
            self.std_pattern = norm_weekly.std(axis=0)    # [points_per_week, D]
        else:
            # 全部按位置计算均值与方差
            self.mean_pattern = weekly.mean(axis=0)  # [points_per_week, D]
            self.std_pattern = weekly.std(axis=0)    # [points_per_week, D]

    # 可视化
    def plot(self, title='Weekly Pattern'):
        D = self.mean_pattern.shape[1]
        for d in range(D):
            plt.figure(figsize=(14, 5))
            plt.plot(self.mean_pattern[:, d], label='Mean Weekly Pattern', color='blue')
            plt.fill_between(range(len(self.mean_pattern)),
                             self.mean_pattern[:, d] - self.std_pattern[:, d],
                             self.mean_pattern[:, d] + self.std_pattern[:, d],
                             color='blue', alpha=0.2, label='±1 Std Dev')

            # Create day labels
            days = ['0', '1', '2', '3', '4', '5', '6', '7']

            # Create x-axis tick positions and labels
            tick_positions = [i * 24 for i in range(8)]  # Start of each day + end
            tick_labels = days

            # Set x-axis ticks and labels
            plt.xticks(tick_positions, tick_labels)
            
            # Add vertical lines to separate days
            for i in range(1, 7):
                plt.axvline(x=i*24, color='gray', linestyle='--', alpha=0.7)
            
            plt.title(title)
            plt.xlabel('Day of Week, dimension {}'.format(d))
            plt.ylabel('Z-score' if self.normalize_by_week else 'Value')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def get_pattern(self, start_idx, end_index):
        """
        [start_idx, end_index) -> [end_index - start_idx, D]
        """
        start_b, start_i = start_idx // self.points_per_week, start_idx % self.points_per_week
        end_b, end_i = end_index // self.points_per_week, end_index % self.points_per_week
        if start_b == end_b:
            return self.mean_pattern[start_i:end_i]
        elif start_b + 1 == end_b:
            pattern = np.concatenate([self.mean_pattern[start_i:], self.mean_pattern[:end_i]], axis=0)
            return pattern
        else:
            repeat = end_b - start_b - 1
            start_part = self.mean_pattern[start_i:]
            middle_part = np.tile(self.mean_pattern, (repeat, 1))
            end_part = self.mean_pattern[:end_i]
            pattern = np.concatenate([start_part, middle_part, end_part], axis=0)
            return pattern

    def transform(self, data, start_idx, end_index):
        """
        data: [L, D]
        L = end_index - start_idx
        """
        assert data.shape[0] == end_index - start_idx
        pat = self.get_pattern(start_idx, end_index)
        return data - pat

    def inverse_transform(self, data, start_idx, end_index):
        """
        data: [L, D]
        L = end_index - start_idx
        """
        assert data.shape[0] == end_index - start_idx
        pat = self.get_pattern(start_idx, end_index)
        return data + pat
    
    def save(self, path):
        np.savez(path, mean_pattern=self.mean_pattern, std_pattern=self.std_pattern)

    def load(self, path):
        data = np.load(path)
        self.mean_pattern = data['mean_pattern']
        self.std_pattern = data['std_pattern']