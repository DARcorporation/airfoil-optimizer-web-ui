import os
import time

if __name__ == '__main__':
    if os.path.isfile(os.environ['RUNFILE']):
        os.remove(os.environ['RUNFILE'])

    time.sleep(5)
    with open(os.environ['RUNFILE'], 'a') as f:
        f.write('1., 3, 3, 8, 8, gen=0, constrain_moment=False, report=False\n')

    with open(os.environ['RUNFILE'], 'a') as f:
        f.write('quit\n')
