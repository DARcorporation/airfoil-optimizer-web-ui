import configparser
import datetime
import os
import shutil
import smtplib
import subprocess
import sys
import zipfile

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

config = configparser.ConfigParser()
config.read('filenames.cfg')
config.read(os.environ['SMTP_SETTINGS'])
config = config['DEFAULT']


def run(cl, n_c, n_t, b_c=8, b_t=8, b_te=8, gen=100,
        fix_te=True,
        constrain_thickness=True, constrain_area=True, constrain_moment=True,
        cm_ref=None, seed=None, n_proc=28,
        run_name=None, report=False):
    """
    Solve the specified optimization problem and handle reporting of results.

    Parameters
    ----------
    cl : float
        Design lift coefficient
    n_c, n_t : int
        Number of CST coefficients for the chord line and thickness distribution, respectively
    b_c, b_t, b_te : int, optional
        Number of bits to encode each of the CST coefficients of the chord line/thickness distribution, and TE thickness
        8 bits each by default.
    gen : int, optional
        Number of generations to use for the genetic algorithm. 100 by default
    fix_te : bool, optional
        True if the trailing edge thickness should be fixed. True by default
    constrain_thickness, constrain_area, constrain_moment : bool, optional
        True if the thickness, area, and/or moment coefficient should be constrained, respectively. All True by default
    cm_ref : float, optional
        If constrain_moment is True, this will be the maximum (absolute) moment coefficient. If None, initial Cm is used
    seed : int, optional
        Seed to use for the random number generator which creates an initial population for the genetic algorithm
    n_proc : int, optional
        Number of processors to use to evaluate functions in parallel using MPI. 28 by default
    run_name : str, optional
        Name of the run. If None, an ISO formatted UTC timestamp will be used.
    report : bool, optional
        True if the results should be reported via email.
    """
    try:
        if run_name is None:
            now = datetime.datetime.utcnow()
            run_name = now.isoformat(timespec='seconds').replace('-', '').replace(':', '') + 'Z'

        path = os.path.join(os.path.abspath(os.environ['RESULTS_DIR']), run_name)
        os.mkdir(path)

        cmd = ['mpirun', '-np', str(n_proc),
               'python3', '-u', 'problem.py',
               str(cl), str(n_c), str(n_t),
               str(b_c), str(b_t), str(b_te),
               str(gen), str(fix_te),
               str(constrain_thickness), str(constrain_area), str(constrain_moment),
               str(cm_ref), str(seed)]

        with open(config['log'], 'wb') as log:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for line in process.stdout:
                sys.stdout.write(line.decode('utf-8'))
                log.write(line)

                if process.poll() is not None:
                    break

        with zipfile.ZipFile(config['sql_zip'], 'w') as zf:
            for i in range(n_proc):
                zf.write(f'{config["sql_base"]}_{i}')

        shutil.copy(config['repr'], path)
        shutil.copy(config['png'], path)
        shutil.copy(config['dat'], path)
        shutil.copy(config['log'], path)
        shutil.copy(config['sql_zip'], path)

        if report:
            msg = MIMEMultipart()
            msg['From'] = config['user']
            msg['To'] = config['receiver']
            msg['Subject'] = 'Airfoil Optimization Complete!'
            with open(config['repr'], 'r') as f:
                msg.attach(MIMEText(f.read(), 'plain'), )
            with open(config['png'], 'rb') as fp:
                attachment = MIMEImage(fp.read(), _subtype='png')
                attachment.add_header('Content-Disposition', 'attachment', filename=config['png'])
                msg.attach(attachment)
            with open(config['dat'], 'r') as f:
                attachment = MIMEText(f.read())
                attachment.add_header('Content-Disposition', 'attachment', filename=config['dat'])
                msg.attach(attachment)
            with open(config['log'], 'r', encoding='utf-8') as f:
                attachment = MIMEText(f.read())
                attachment.add_header('Content-Disposition', 'attachment', filename=config['log'])
                msg.attach(attachment)

            with smtplib.SMTP_SSL(config['host'], int(config['port'])) as server:
                server.ehlo()
                server.login(config['user'], config['password'])
                server.sendmail(config['user'], config['receiver'], msg.as_string())
                print('Email sent')
    except Exception as e:
        print(e)


def main():
    """
    Run all optimization problems defined in the Runfile.
    """
    with open('Runfile', 'r') as f:
        eval(f'run({f.readline()})')


if __name__ == '__main__':
    main()
