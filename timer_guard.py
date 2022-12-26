import os
import pdb
import time

class TimerGuard:
    def __init__(self, name="", group="KGRL_TIMER_ENABLE", text="{}: cost {} ms", logger=print):
        self.name = name
        self.group = group
        self.text = text
        self.logger = logger

        self.start_time = None
        self.start()

    def start(self):
        """Start a new timer"""
        if os.getenv(self.group) == "True":
            #if self.start_time is not None:
            #    raise TimerError(f"Timer is running. Use .stop() to stop it")

            self.start_time = time.time()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if os.getenv(self.group) == "True":
            if self.start_time is None:
                #raise TimerError(f"Timer is not running. Use .start() to start it")
                return

            elapsed_time = time.time() - self.start_time
            self.start_time = None

            if self.logger:
                self.logger(self.text.format(self.name, elapsed_time*1000))

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the context manager timer"""
        self.stop()

    def __call__(self, fn):
        def warp_fn(*args, **kwargs):
            if os.getenv(self.group) == "True":
                start = time.time()
            result = fn(*args, **kwargs)
            if os.getenv(self.group) == "True":
                elapsed_time = time.time() - start
                if self.logger:
                    self.logger(self.text.format(self.name, elapsed_time*1000))
            return result
        return warp_fn

    def __del__(self):
        self.stop()

if __name__ == '__main__':
    group = 'KGRL_TIMER_ENABLE'
    os.environ[group] = "True"

    # As a class (start/stop)
    t1 = TimerGuard("test1", group)
    t1.start()
    time.sleep(1)
    t1.stop()

    # As a class (init/del)
    t2 = TimerGuard("test2", group)
    time.sleep(1)
    del t2

    # As a context manager
    with TimerGuard("test3", group):
        time.sleep(1)

    # As a decorator
    @TimerGuard("test4", group)
    def sleep(sec):
        time.sleep(sec)

    sleep(1)
