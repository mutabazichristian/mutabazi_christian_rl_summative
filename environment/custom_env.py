import random
from enum import Enum

import gymnasium as gym
import numpy as np


class TaskType(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    BASIC = "basic"


class Task:
    def __init__(self, task_type, assigned_time):
        self.type = task_type
        self.assigned_time = assigned_time

        if task_type == TaskType.HIGH:
            self.duration = 30
            self.reward = 10
            self.loss = 10
            self.loss_late = 20
            self.window = 50

        elif task_type == TaskType.MEDIUM:
            self.duration = 10
            self.reward = 5
            self.loss = 5
            self.loss_late = 10
            self.window = 60

        else:
            self.duration = 5
            self.reward = 2
            self.loss = 2
            self.loss_late = 5
            self.window = 120

        self.deadline = assigned_time + self.window
        self.progress = 0
        self.picked_up = False
        self.completed = False


class WorkplaceEnv(gym.Env):
    def __init__(self, render_mode=None):

        super().__init__()
        self.TOTAL_MINUTES = 480
        self.MAX_WORKING_TASKS = 3
        self.STARTING_TRUST = 100
        self.HOURLY_BONUS = 30

        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0, 0]),
            high=np.array([480, 200, 3, 10, 1]),
            dtype=np.float32,
        )

        self.action_space = gym.spaces.Discrete(6)

        self.render_mode = render_mode
        self.reset()



    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.current_time = 0
        self.trust_points = self.STARTING_TRUST
        self.active_tasks = []
        self.available_tasks = []
        self.failed_tasks = []
        self.completed_tasks = []
        self.last_hourly_bonus = 0

        self._generate_random_tasks()

        return self._get_observation(), {}

    def _generate_random_tasks(self):
        if random.random() < 0.3:
            task_type = random.choices(
                [TaskType.HIGH, TaskType.MEDIUM, TaskType.BASIC],
                weights=[0.2, 0.3, 0.5],
            )[0]

            task = Task(task_type, self.current_time)
            self.available_tasks.append(task)

    def _get_observation(self):
        time_norm = self.current_time / self.TOTAL_MINUTES

        trust_norm = self.trust_points / self.STARTING_TRUST

        num_active = len(self.active_tasks)
        num_avaialbe = min(len(self.available_tasks), 10)
        next_urgency = 0

        if self.available_tasks:
            next_task = self.available_tasks[0]
            time_left = next_task.deadline - self.current_time
            next_urgency = max(0, 1 - (time_left / next_task.window))

        return np.array(
            [time_norm, trust_norm, num_active, num_avaialbe, next_urgency],
            dtype=np.float32,
        )

    def step(self, action):
        reward = 0
        terminated = False

        if action == 0:
            pass
        elif action == 1:
            reward += self._pick_up_task()

        elif action >= 2 and action <= 4:
            task_index = action - 2
            reward += self._work_on_task(task_index)

        self.current_time += 1

        self._generate_random_tasks()

        reward += self._check_deadlines()

        if (
            self.current_time % 60 == 0
            and self.current_time > self.last_hourly_bonus
        ):
            self.trust_points += self.HOURLY_BONUS
            reward += 5
            self.last_hourly_bonus = self.current_time

        if self.trust_points <= 0:
            terminated = True
            reward -= 50
        elif self.current_time >= self.TOTAL_MINUTES:
            terminated = True
            if self.trust_points > 0:
                reward += 50
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _pick_up_task(self):
        if (
            not self.available_tasks
            or len(self.active_tasks) >= self.MAX_WORKING_TASKS
        ):
            return -1

        task = self.available_tasks.pop(0)
        task.picked_up = True
        self.active_tasks.append(task)
        return 1

    def _work_on_task(self, task_index):

        if task_index >= len(self.active_tasks):
            return -1

        task = self.active_tasks[task_index]
        task.progress += 1

        if task.progress >= task.duration:
            task.completed = True
            self.trust_points += task.reward
            reward = task.reward
            self.completed_tasks.append(task)
            self.active_tasks.remove(task)
            return reward

        return 0.1

    def _check_deadlines(self):
        reward = 0

        expired_available = []
        for task in self.available_tasks:
            if self.current_time >= task.deadline:
                expired_available.append(task)

        for task in expired_available:
            self.available_tasks.remove(task)
            self.trust_points -= task.loss
            reward -= task.loss
            self.failed_tasks.append(task)

        expired_active = []
        for task in self.active_tasks:
            if self.current_time >= task.deadline:
                expired_active.append(task)

        for task in expired_active:
            self.active_tasks.remove(task)
            self.trust_points -= task.loss_late
            reward -= task.loss_late
            self.failed_tasks.append(task)

        return reward

    def _get_info(self):
        return {
            "trust_points": self.trust_points,
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "active_tasks": len(self.active_tasks),
            "available_tasks": len(self.available_tasks),
            "time_left": self.TOTAL_MINUTES - self.current_time,
        }

    def render(self):
        if self.render_mode == "human":
            print(
                f"Time: {self.current_time:3d}/480 | Trust: {self.trust_points:3d} | "
                f"Active: {len(self.active_tasks)} | Available: {len(self.available_tasks)} | "
                f"Done: {len(self.completed_tasks)} | Failed: {len(self.failed_tasks)}"
            )
