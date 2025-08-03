import pygame
import math


class GameVisualization:

    def __init__(self, env):
        pygame.init()
        self.env = env
        self.width, self.height = 1400, 900
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Workplace Agent - Survival Mode")


        self.COLORS = {
            'background': (245, 248, 252),
            'primary': (59, 130, 246),
            'success': (34, 197, 94),
            'warning': (251, 191, 36),
            'danger': (239, 68, 68),
            'secondary': (107, 114, 128),
            'white': (255, 255, 255),
            'black': (17, 24, 39),
            'gray_light': (243, 244, 246),
            'gray': (156, 163, 175),
            'blue_light': (147, 197, 253),
            'green_light': (134, 239, 172),
            'red_light': (252, 165, 165),
            'purple': (147, 51, 234),
            'orange': (249, 115, 22)
        }

        # Task type colors
        self.TASK_COLORS = {
            'HIGH': self.COLORS['danger'],
            'MEDIUM': self.COLORS['warning'],
            'BASIC': self.COLORS['success']
        }

        # Fonts
        self.fonts = {
            'title': pygame.font.Font(None, 48),
            'large': pygame.font.Font(None, 36),
            'medium': pygame.font.Font(None, 24),
            'small': pygame.font.Font(None, 18),
            'tiny': pygame.font.Font(None, 14)
        }

        # Layout constants
        self.MARGIN = 30
        self.CARD_RADIUS = 12
        self.SHADOW_OFFSET = 3

    def draw_rounded_rect(self, surface, color, rect, radius=12, shadow=True):
        if shadow:
            shadow_rect = pygame.Rect(rect.x + self.SHADOW_OFFSET, rect.y + self.SHADOW_OFFSET,
                                    rect.width, rect.height)
            pygame.draw.rect(surface, (0, 0, 0, 30), shadow_rect, border_radius=radius)

        pygame.draw.rect(surface, color, rect, border_radius=radius)

    def draw_progress_bar(self, surface, x, y, width, height, progress, bg_color, fill_color, text=""):
        bg_rect = pygame.Rect(x, y, width, height)
        self.draw_rounded_rect(surface, bg_color, bg_rect, radius=height//2, shadow=False)

        if progress > 0:
            fill_width = max(height, int(width * progress))
            fill_rect = pygame.Rect(x, y, fill_width, height)
            self.draw_rounded_rect(surface, fill_color, fill_rect, radius=height//2, shadow=False)

        if text:
            text_surface = self.fonts['small'].render(text, True, self.COLORS['white'])
            text_rect = text_surface.get_rect(center=(x + width//2, y + height//2))
            surface.blit(text_surface, text_rect)

    def draw_header(self):
        header_rect = pygame.Rect(0, 0, self.width, 120)
        self.draw_rounded_rect(self.screen, self.COLORS['primary'], header_rect, radius=0, shadow=False)

        for i in range(120):
            alpha = int(255 * (1 - i/120) * 0.3)
            color = (*self.COLORS['blue_light'], alpha)
            pygame.draw.line(self.screen, color, (0, i), (self.width, i))

        title_text = self.fonts['title'].render("Workplace Survival", True, self.COLORS['white'])
        self.screen.blit(title_text, (self.MARGIN, 20))

        time_progress = self.env.current_time / self.env.TOTAL_MINUTES
        time_x = self.width - 350
        time_y = 25

        time_label = self.fonts['medium'].render("Work Day Progress", True, self.COLORS['white'])
        self.screen.blit(time_label, (time_x, time_y))

        self.draw_progress_bar(
            self.screen, time_x, time_y + 30, 300, 25, time_progress,
            self.COLORS['blue_light'], self.COLORS['green_light'],
            f"{self.env.current_time}m / 480m"
        )

        remaining = 480 - self.env.current_time
        remaining_text = self.fonts['small'].render(f"{remaining} minutes remaining", True, self.COLORS['white'])
        self.screen.blit(remaining_text, (time_x, time_y + 65))

    def draw_trust_meter(self):
        x, y = self.MARGIN, 140
        width, height = 300, 120

        card_rect = pygame.Rect(x, y, width, height)
        self.draw_rounded_rect(self.screen, self.COLORS['white'], card_rect)

        title = self.fonts['large'].render("Trust Level", True, self.COLORS['black'])
        self.screen.blit(title, (x + 20, y + 15))

        trust_color = self.COLORS['success'] if self.env.trust_points > 50 else (
            self.COLORS['warning'] if self.env.trust_points > 20 else self.COLORS['danger']
        )
        trust_text = self.fonts['large'].render(f"{self.env.trust_points}", True, trust_color)
        self.screen.blit(trust_text, (x + 20, y + 50))

        trust_progress = max(0, self.env.trust_points / 100)
        bar_color = self.COLORS['success'] if trust_progress > 0.5 else (
            self.COLORS['warning'] if trust_progress > 0.2 else self.COLORS['danger']
        )
        self.draw_progress_bar(
            self.screen, x + 120, y + 55, 150, 20, trust_progress,
            self.COLORS['gray_light'], bar_color
        )

        status = "Excellent" if self.env.trust_points > 70 else (
            "Good" if self.env.trust_points > 40 else (
                "Warning" if self.env.trust_points > 0 else "Critical"
            )
        )
        status_text = self.fonts['small'].render(f"Status: {status}", True, self.COLORS['secondary'])
        self.screen.blit(status_text, (x + 20, y + 85))

    def draw_active_tasks(self):
        x, y = self.MARGIN, 280
        width = 550

        title = self.fonts['large'].render("Active Tasks", True, self.COLORS['black'])
        self.screen.blit(title, (x, y))
        y += 40

        if not self.env.active_tasks:
            no_tasks_rect = pygame.Rect(x, y, width, 60)
            self.draw_rounded_rect(self.screen, self.COLORS['gray_light'], no_tasks_rect)

            no_tasks_text = self.fonts['medium'].render("No active tasks", True, self.COLORS['secondary'])
            text_rect = no_tasks_text.get_rect(center=no_tasks_rect.center)
            self.screen.blit(no_tasks_text, text_rect)
        else:
            for i, task in enumerate(self.env.active_tasks[:3]):
                self.draw_task_card(x, y + i * 90, width, 80, task, active=True)

    def draw_available_tasks(self):
        x, y = 620, 280
        width = 350

        title = self.fonts['large'].render("Available Tasks", True, self.COLORS['black'])
        self.screen.blit(title, (x, y))
        y += 40

        if not self.env.available_tasks:
            no_tasks_rect = pygame.Rect(x, y, width, 60)
            self.draw_rounded_rect(self.screen, self.COLORS['gray_light'], no_tasks_rect)

            no_tasks_text = self.fonts['medium'].render("No new tasks", True, self.COLORS['secondary'])
            text_rect = no_tasks_text.get_rect(center=no_tasks_rect.center)
            self.screen.blit(no_tasks_text, text_rect)
        else:
            for i, task in enumerate(self.env.available_tasks[:5]):
                self.draw_task_card(x, y + i * 65, width, 55, task, active=False)

    def draw_task_card(self, x, y, width, height, task, active=True):
        task_type = task.type.value.upper()
        color = self.TASK_COLORS.get(task_type, self.COLORS['secondary'])

        card_rect = pygame.Rect(x, y, width, height)
        self.draw_rounded_rect(self.screen, self.COLORS['white'], card_rect)

        indicator_rect = pygame.Rect(x, y, 8, height)
        self.draw_rounded_rect(self.screen, color, indicator_rect, radius=self.CARD_RADIUS, shadow=False)

        badge_width = 80
        badge_rect = pygame.Rect(x + width - badge_width - 10, y + 8, badge_width, 20)
        self.draw_rounded_rect(self.screen, color, badge_rect, radius=10, shadow=False)

        badge_text = self.fonts['tiny'].render(task_type, True, self.COLORS['white'])
        badge_text_rect = badge_text.get_rect(center=badge_rect.center)
        self.screen.blit(badge_text, badge_text_rect)

        text_x = x + 20

        if active:
            progress_text = self.fonts['medium'].render(f"Progress: {task.progress}/{task.duration}", True, self.COLORS['black'])
            self.screen.blit(progress_text, (text_x, y + 10))

            progress = task.progress / task.duration if task.duration > 0 else 0
            self.draw_progress_bar(
                self.screen, text_x, y + 35, width - 120, 15, progress,
                self.COLORS['gray_light'], color
            )

            deadline_text = self.fonts['small'].render(f"Due: {task.deadline}m", True, self.COLORS['secondary'])
            self.screen.blit(deadline_text, (text_x, y + 55))
        else:
            duration_text = self.fonts['medium'].render(f"Duration: {task.duration}m", True, self.COLORS['black'])
            self.screen.blit(duration_text, (text_x, y + 8))

            deadline_text = self.fonts['small'].render(f"Deadline: {task.deadline}m", True, self.COLORS['secondary'])
            self.screen.blit(deadline_text, (text_x, y + 28))

    def draw_statistics(self):
        x, y = self.MARGIN, 650

        stats = [
            ("Completed", len(self.env.completed_tasks), self.COLORS['success']),
            ("Failed", len(self.env.failed_tasks), self.COLORS['danger']),
            ("Success Rate", f"{len(self.env.completed_tasks)/(max(1, len(self.env.completed_tasks) + len(self.env.failed_tasks)))*100:.0f}%", self.COLORS['primary'])
        ]

        card_width = 180
        for i, (label, value, color) in enumerate(stats):
            card_x = x + i * (card_width + 20)
            card_rect = pygame.Rect(card_x, y, card_width, 100)
            self.draw_rounded_rect(self.screen, self.COLORS['white'], card_rect)

            label_text = self.fonts['medium'].render(label, True, self.COLORS['secondary'])
            self.screen.blit(label_text, (card_x + 15, y + 15))

            value_text = self.fonts['large'].render(str(value), True, color)
            self.screen.blit(value_text, (card_x + 15, y + 45))

    def draw_action_legend(self):
        x, y = 620, 650
        width, height = 350, 200

        card_rect = pygame.Rect(x, y, width, height)
        self.draw_rounded_rect(self.screen, self.COLORS['white'], card_rect)

        title = self.fonts['large'].render("Controls", True, self.COLORS['black'])
        self.screen.blit(title, (x + 20, y + 15))

        actions = [
            "0: Wait/Do Nothing",
            "1: Pick Up Next Task",
            "2-4: Work on Task 1-3",
            "",
            "SPACE: Random Action",
            "ESC: Quit Demo"
        ]

        for i, action in enumerate(actions):
            if action:
                action_text = self.fonts['small'].render(action, True, self.COLORS['secondary'])
                self.screen.blit(action_text, (x + 20, y + 50 + i * 20))

    def render(self):
        self.screen.fill(self.COLORS['background'])

        self.draw_header()
        self.draw_trust_meter()
        self.draw_active_tasks()
        self.draw_available_tasks()
        self.draw_statistics()
        self.draw_action_legend()

        pygame.display.flip()

    def close(self):
        pygame.quit()
