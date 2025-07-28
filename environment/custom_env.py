import gymnasium as gym

class TaskType(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    BASIC = "basic"

class Task:
    def __init__(self,task_type,assigned_time):
        self.type = task_typeA
        self.assigned_time = assigned_time

        if task_type = TaskType.HIGH:
            self.duration = 30
            self.reward = 10
            self.loss = 10
            self.loss_late=20
            self.window = 50
            # self.window = random.randint(40,60)

        elif task_type = TaskType.MEDIUM:
            self.duration = 10
            self.reward = 5
            self.loss = 5
            self.loss_late = 10
            self.window=60
            # self.window=random.randint(30,90)

        else :
            self.duration = 5
            self.reward =2
            self.loss=2
            self.late=5
            self.window=120
            # self.window= random.randint(60,180)
            pass

        self.deadline = assigned_time +self.window
        self.progress = 0
        self.picked_up = False
        self.completed = False

class WorkplaceEnv(gym.Env):
    def __init__(self,render_mode=None):
        super().__init__()
        self.TOTAL_MINUTES = 480
        self.MAX_WORKING_TASKS = 3
        self.STARTING_TRUST=100
        self.HOURLY_BONUS=30

        self.observation=spaces.Box(
                low=
                )

    def hi(self):
        print("hellow")

    def reset(self):
        self.trust_points = 100
        self.minutes_left = 0

    def render_print(self):
        print("stuff")
    def generate task(self)
    #


work = WorkplaceEnv()
work.hi()
