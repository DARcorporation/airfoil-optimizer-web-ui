import os
import time

if __name__ == "__main__":
    if os.path.isfile(os.environ["RUNFILE"]):
        os.remove(os.environ["RUNFILE"])

    time.sleep(5)
    with open(os.environ["RUNFILE"], "a") as f:
        for _ in range(4):
            f.write("1., 3, 3, 16, 16, gen=300, constrain_moment=False, report=True\n")

        for _ in range(4):
            f.write(".5, 3, 3, 16, 16, gen=300, constrain_moment=False, report=True\n")

        f.write("quit\n")
