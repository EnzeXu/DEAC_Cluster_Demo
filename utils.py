import time
import torch
import argparse


def add_time(func):
    def wrapper(*args, **kwargs):
        if func.__name__ == "myprint":
            with open(args[1], "a") as f:
                f.write("{} ".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))))
        print("[{}] ".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))), end="")
        ret = func(*args, **kwargs)
        return ret
    return wrapper


@add_time
def myprint(string, filename):
    with open(filename, "a") as f:
        f.write("{}\n".format(string))
    print(string)


def demo():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=100000, help="epoch")
    parser.add_argument("--log_path", type=str, default="logs/1.txt", help="log path")
    # parser.add_argument("--epoch_step", type=int, default=1000, help="epoch_step")
    # parser.add_argument('--lr', type=float, default=0.01, help='learning rate, default=0.001')
    # parser.add_argument("--main_path", default=".", help="main_path")
    # parser.add_argument("--save_step", type=int, default=10000, help="save_step")
    opt = parser.parse_args()

    log_path = opt.log_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch = opt.epoch
    myprint("log_path: {}".format(log_path), log_path)
    myprint("cuda is available: {}".format(torch.cuda.is_available()), log_path)
    myprint("using: {}".format(device), log_path)
    myprint("epoch: {}".format(epoch), log_path)


if __name__ == "__main__":
    # myprint("hello world!", "logs/1.txt")
    print(torch.cuda.is_available())
    pass
