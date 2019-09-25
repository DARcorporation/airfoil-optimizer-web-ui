import configparser
import datetime
import os
import requests
import smtplib
import subprocess
import sys
import time

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

config = configparser.ConfigParser()
config.read(os.environ["SMTP_SETTINGS"])
config = config["DEFAULT"]


def run(
    cl,
    n_c,
    n_t,
    gen=100,
    tolx=1e-8,
    tolf=1e-8,
    fix_te=True,
    t_te_min=0.,
    t_c_min=0.01,
    A_cs_min=None,
    Cm_max=None,
    seed=None,
    n_proc=28,
    run_name=None,
    report=False,
):
    """
    Solve the specified optimization problem and handle reporting of results.

    Parameters
    ----------
    cl : float
        Design lift coefficient
    n_c, n_t : int
        Number of CST coefficients for the chord line and thickness distribution, respectively
    gen : int, optional
        Number of generations to use for the genetic algorithm. 100 by default
    fix_te : bool, optional
        True if the trailing edge thickness should be fixed. True by default
    t_te_min : float, optional
        Minimum TE thickness as fraction of chord length. Default is 0.0.
    t_c_min : float or None, optional
        Minimum thickness over chord ratio. None if unconstrained. Defaults is 0.01.
    A_cs_min : float or None, optional
        Minimum cross sectional area. None if unconstrained. Default is None.
    Cm_max : float or None, optional
        Maximum absolute moment coefficient. None if unconstrained. Default is None.
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
            run_name = (
                now.isoformat(timespec="seconds").replace("-", "").replace(":", "")
                + "Z"
            )

        path = os.path.join(os.path.abspath(os.environ["RESULTS_DIR"]), run_name)
        os.makedirs(path)

        repr_file = os.path.join(path, "repr.yml")
        dat_file = os.path.join(path, "optimized.dat")
        png_file = os.path.join(path, "optimized.png")
        log_file = os.path.join(path, "log.txt")

        cmd = [
            "mpirun",
            "-np",
            str(n_proc),
            "python3",
            "-u",
            "-m",
            "af_opt.problem",
            str(cl),
            str(n_c),
            str(n_t),
            str(gen),
            str(tolx),
            str(tolf),
            str(fix_te),
            str(t_te_min),
            str(t_c_min),
            str(A_cs_min),
            str(Cm_max),
            str(seed),
            str(repr_file),
            str(dat_file),
            str(png_file),
        ]
        print(f"Going to run the following command: \n{cmd}")

        with open(log_file, "wb") as log:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )
            for line in process.stdout:
                sys.stdout.write(line.decode("utf-8"))
                log.write(line)

                if process.poll() is not None:
                    break

        if report:
            print("Going to send an email")

            msg = MIMEMultipart()
            msg["From"] = config["user"]
            msg["To"] = config["receiver"]
            msg["Subject"] = "Airfoil Optimization Complete!"
            with open(repr_file, "r") as f:
                msg.attach(MIMEText(f.read(), "plain"))

                f.seek(0)
                attachment = MIMEText(f.read(), _subtype="yaml")
                attachment.add_header(
                    "Content-Disposition",
                    "attachment",
                    filename=os.path.basename(repr_file),
                )
                msg.attach(attachment)
            with open(png_file, "rb") as fp:
                attachment = MIMEImage(fp.read(), _subtype="png")
                attachment.add_header(
                    "Content-Disposition",
                    "attachment",
                    filename=os.path.basename(png_file),
                )
                msg.attach(attachment)
            with open(dat_file, "r") as f:
                attachment = MIMEText(f.read())
                attachment.add_header(
                    "Content-Disposition",
                    "attachment",
                    filename=os.path.basename(dat_file),
                )
                msg.attach(attachment)
            with open(log_file, "r", encoding="utf-8") as f:
                attachment = MIMEText(f.read())
                attachment.add_header(
                    "Content-Disposition",
                    "attachment",
                    filename=os.path.basename(log_file),
                )
                msg.attach(attachment)

            with smtplib.SMTP_SSL(config["host"], int(config["port"])) as server:
                server.ehlo()
                server.login(config["user"], config["password"])
                server.sendmail(config["user"], config["receiver"], msg.as_string())
                print("Email sent")
    except Exception as e:
        print(e)


def main():
    """
    Poll runs service for new run cases and run them.
    """
    host = os.environ["RUNS_SERVICE_URL"]

    while True:
        r = requests.get(f"{host}/runs/accept")
        if r.status_code == 204:
            time.sleep(1)
            continue

        response_object = r.json()
        id = response_object["data"]["id"]

        kwargs = dict(response_object["data"])
        del kwargs["id"]
        del kwargs["status"]
        print(f"Got a request to start a run with the following data: \n{kwargs}")

        try:
            run(**kwargs)
            requests.post(f"{host}/runs/complete", json={"id": id, "success": True})
        except TypeError:
            print("Invalid run case")
            requests.post(f"{host}/runs/complete", json={"id": id, "success": False})


if __name__ == "__main__":
    main()
