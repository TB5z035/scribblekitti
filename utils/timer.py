import time
from datetime import datetime


class Timer:

    def __init__(self, name=None, slient=False, log_fn=print):
        self.ticks = []
        self.name = 'default' if name is None else name
        self.log_fn = log_fn if not slient else (lambda *args, **kwargs: None)

    def __enter__(self):
        self.ticks.append(datetime.now())
        return self

    def tick(self, name=None):
        self.ticks.append(datetime.now())
        self.log_fn(f'[{self.name:15s}] {("[" + name + "] ") if name is not None else "":25s}Time elapsed: {self.ticks[-1] - self.ticks[-2]}')

    def __exit__(self, *args):
        self.ticks.append(datetime.now())
        self.deltas = [self.ticks[i + 1] - self.ticks[i] for i in range(len(self.ticks) - 1)]
        self.total = sum(self.deltas, datetime.min - datetime.min)
        self.total_seconds = self.total.total_seconds()
        self.log_fn(f'[{self.name:15s}] {"Total":25s}Time elapsed: {self.total} ({self.total_seconds} seconds))')


def test_timer():
    with Timer() as timer:
        timer.tick()
        time.sleep(1)
        timer.tick('sleep 1')
        timer.tick()


if __name__ == '__main__':
    test_timer()