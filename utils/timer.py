import time
import datetime

class contexttimer:
    def __enter__(self):
        self.start = time.time()
        return self
        
    def __exit__(self, typ, value, tb):
        self.seconds = time.time() - self.start
        self.time_str = datetime.timedelta(seconds=round(self.seconds))


def now():
    return datetime.datetime.now().strftime('%b/%d/%Y, %H:%M:%S')


def tic():
    return time.time()


def today():
    return datetime.date.today().strftime('%b%d')


def sec2str(seconds):
    return datetime.timedelta(seconds=seconds)


if __name__ == "__main__":
    print(f'today: {today()}, now: {now()}')
    with contexttimer() as t:
        time.sleep(0.234)
    print(t.seconds, t.time_str)
    print(sec2str(time.time()))