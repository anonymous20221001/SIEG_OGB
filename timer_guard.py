import os
import time
import logging

logger = logging.getLogger()


class TimerGuard:
    def __init__(self, name="", group="KGRL_TIMER_ENABLE"):
        self.name = name
        self.start = 0.
        self.group = group

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if os.getenv(self.group) == "True":
            #logger.info(f"{self.name}: cost {(time.time() - self.start) * 1000} ms")
            print(f"{self.name}: cost {(time.time() - self.start) * 1000} ms")

    def __call__(self, fn):
        def warp_fn(*args, **kwargs):
            start = time.time()
            result = fn(*args, **kwargs)
            if os.getenv(self.group) == "True":
                #logger.info(f"{self.name}: cost {(time.time() - start) * 1000} ms")
                print(f"{self.name}: cost {(time.time() - start) * 1000} ms")
            return result
        return warp_fn