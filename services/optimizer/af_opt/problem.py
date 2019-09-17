import numpy as np
import openmdao.api as om
import os
import sys
import time

from datetime import timedelta
from multiprocessing import TimeoutError
from multiprocessing.pool import ThreadPool
from random import SystemRandom
from xfoil import XFoil
from xfoil.model import Airfoil

from .cst import cst, fit
from .util import cosspace
from .genetic_algorithm_driver import SimpleGADriver

# Ensure MPI is defined
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

# Detect whether the script is being run under MPI and grab the rank
if not MPI:
    run_parallel = False
    rank = 0
    n_proc = 1
else:
    run_parallel = True
    rank = MPI.COMM_WORLD.rank
    n_proc = MPI.COMM_WORLD.size

# Numpy string formatters
array_formatter = {"float_kind": "{: 7.4f}".format}

# Reference airfoil coordinates
file_path = os.path.abspath(os.path.dirname(__file__))
coords_file = os.path.join(file_path, "naca0012.dat")
coords_ref = np.loadtxt(coords_file, skiprows=1)

# Reference airfoil coordinates split between upper and lower surfaces
i_0_ref = np.argmin(coords_ref[:, 0])
coords_ref_u = np.flipud(coords_ref[: i_0_ref + 1, :])
coords_ref_l = coords_ref[i_0_ref:, :]
t_te_ref = coords_ref_u[-1, 1] - coords_ref_l[-1, 1]

# Upper and lower reference airfoil y-coordinates sampled at identical x-coordinates
x_ref = coords_ref_u[:, 0]
y_u_ref = coords_ref_u[:, 1]
y_l_ref = np.interp(x_ref, coords_ref_l[:, 0], coords_ref_l[:, 1])


def coords2cst(x, y_u, y_l, n_c, n_t):
    """
    Convert airfoil upper/lower curve coordinates to camber line/thickness distribution CST coefficients.

    Parameters
    ----------
    x : array_like
        X-Coordinates
    y_u, y_l : array_like
        Y-Coordinates of the upper and lower curves, respectively
    n_c, n_t : int
        Number of CST coefficients to use for the camber line and thickness distribution of the airfoil

    Returns
    -------
    a_c, a_t : np.ndarray
        CST coefficients describing the camber line and thickness distribution of the airfoil
    t_te : float
        Airfoil trailing edge thickness
    """
    y_c = (y_u + y_l) / 2
    t = y_u - y_l

    a_c, _ = fit(x, y_c, n_c, delta=(0.0, 0.0), n1=1)
    a_t, t_te = fit(x, t, n_t)

    return a_c, a_t, t_te[1]


def cst2coords(a_c, a_t, t_te, n_coords=100):
    """
    Convert airfoil camber line/thickness distribution CST coefficients to upper/lower curve coordinates.

    Parameters
    ----------
    a_c, a_t : array_like
        CST coefficients describing the camber line and thickness distribution of the airfoil
    t_te : float
        Airfoil trailing edge thickness
    n_coords : int, optional
        Number of x-coordinates to use. 100 by default

    Returns
    -------
    x : np.ndarray
        Airfoil x-coordinates
    y_u, y_l : np.ndarray
        Airfoil upper and lower curves y-coordinates
    y_c, t : np.ndarray
        Airfoil camber line and thickness distribution
    """
    x = cosspace(0, 1, n_coords)
    y_c = cst(x, a_c, n1=1)
    t = cst(x, a_t, delta=(0, t_te))

    y_u = y_c + t / 2
    y_l = y_c - t / 2
    return x, y_u, y_l, y_c, t


def xfoil_worker(xf, cl_spec, consistency_check=True, consistency_tol=1e-4):
    """
    Try to operate the given XFoil instance at a specified lift coefficient.

    Parameters
    ----------
    xf : XFoil
        Instance of XFoil class with Airfoil already specified
    cl_spec : float
        Lift coefficient
    consistency_check : bool, optional
        If True, airfoil will be analyzed at least twice to ensure consistent results. True by default.
        This option will run the same airfoil at the same point twice and checks if the results match. If they don't, it
        will be run a third time. It is expected that two out of three results will agree. The third will be considered
        incorrect and discarded. If the first run returns NaN, the airfoil will be assumed unrealistic and it will not
        be run a second time.
    consistency_tol : float, optional
        Tolerance used for the consistency check. 1e-4 by default

    Returns
    -------
    cd, cm : float or np.nan
        Drag and moment coefficients or nan if analysis did not complete successfully
    """
    _, cd1, cm1, _ = xf.cl(cl_spec)
    if np.isnan(cd1) or not consistency_check:
        return cd1, cm1

    xf.reset_bls()
    _, cd2, cm2, _ = xf.cl(cl_spec)

    e = np.abs(cd2 - cd1)
    if e < consistency_tol:
        return cd1, cm1

    xf.reset_bls()
    _, cd3, cm3, _ = xf.cl(cl_spec)

    if np.abs(cd3 - cd1) < consistency_tol:
        return cd1, cm1

    if np.abs(cd3 - cd2) < consistency_tol:
        return cd2, cm2

    return (cd1 + cd2 + cd3) / 3.0, (cm1 + cm2 + cm3) / 3.0


def analyze_airfoil(
    x, y_u, y_l, cl, rey, mach=0, xf=None, pool=None, show_output=False
):
    """
    Analyze an airfoil at a given lift coefficient for given Reynolds and Mach numbers using XFoil.

    Parameters
    ----------
    x : array_like
        Airfoil x-coordinates
    y_u, y_l : array_like
        Airfoil upper and lower curve y-coordinates
    cl : float
        Target lift coefficient
    rey, mach : float
        Reynolds and Mach numbers
    xf : XFoil, optional
        An instance of the XFoil class to use to perform the analysis. Will be created if not given
    pool : multiprocessing.ThreadPool, optional
        An instance of the multiprocessing.Threadpool class used to run the xfoil_worker. Will be created if not given
    show_output : bool, optional
        If True, a debug string will be printed after analyses. False by default.

    Returns
    -------
    cd, cm : float or np.nan
        Drag and moment coefficients of the airfoil at specified conditions, or nan if XFoil did not run successfully
    """
    # If the lower and upper curves swap, this is a bad, self-intersecting airfoil. Return 1e27 immediately.
    if np.any(y_l > y_u):
        return np.nan
    else:
        clean_xf = False
        if xf is None:
            xf = XFoil()
            xf.print = show_output
            clean_xf = True

        clean_pool = False
        if pool is None:
            pool = ThreadPool(processes=1)
            clean_pool = True

        xf.airfoil = Airfoil(
            x=np.concatenate((x[-1:0:-1], x)), y=np.concatenate((y_u[-1:0:-1], y_l))
        )
        # xf.repanel(n_nodes=300, cv_par=2.0, cte_ratio=0.5)
        xf.repanel(n_nodes=240)
        xf.Re = rey
        xf.M = mach
        xf.max_iter = 200

        cd = np.nan
        cm = np.nan
        future = pool.apply_async(xfoil_worker, args=(xf, cl))
        try:
            cd, cm = future.get(timeout=5.0)
            xf.reset_bls()
        except TimeoutError:
            pass

    if clean_xf:
        del xf
    if clean_pool:
        del pool

    return cd, cm


class AirfoilComponent(om.ExplicitComponent):
    """
    Basic Aifoil specified by CST coefficients for its camber line and thickness distribution and a TE thickness.
    """

    def initialize(self):
        self.options.declare("n_c", default=6, types=int)
        self.options.declare("n_t", default=6, types=int)

        self.options.declare("n_coords", default=100, types=int)

    def setup(self):
        # Number of CST coefficients
        n_c = self.options["n_c"]
        n_t = self.options["n_t"]

        # Inputs
        self.add_input("a_c", shape=n_c)
        self.add_input("a_t", shape=n_t)
        self.add_input("t_te", shape=1)

    def compute_coords(self, inputs, precision=None):
        """
        Compute airfoil coordinates from the set of OpenMDAO inputs.
        """
        x, y_u, y_l, y_c, t = cst2coords(
            inputs["a_c"], inputs["a_t"], inputs["t_te"][0], self.options["n_coords"]
        )
        if precision is not None:
            return (
                np.round(x, precision),
                np.round(y_u, precision),
                np.round(y_l, precision),
                np.round(y_c, precision),
                np.round(t, precision),
            )
        return x, y_u, y_l, y_c, t


class XFoilComp(AirfoilComponent):
    """
    Computes the drag coefficient of an airfoil at a given lift coefficient, Reynolds nr., and Mach nr.
    """

    def initialize(self):
        super().initialize()
        self.options.declare("print", default=False, types=bool)

        xf = XFoil()
        xf.print = False
        self.options.declare("_xf", default=xf, types=XFoil, allow_none=True)
        self.options.declare(
            "_pool", default=ThreadPool(processes=1), types=ThreadPool, allow_none=True
        )

        self.recording_options["options_excludes"] = ["_xf", "_pool"]

    def setup(self):
        super().setup()

        # Inputs
        self.add_input("Cl_des", val=1.0)
        self.add_input("Re", val=1e6)
        self.add_input("M", val=0.0)

        # Output
        self.add_output("Cd", val=1.0)
        self.add_output("Cm", val=1.0)

    def compute(self, inputs, outputs, **kwargs):
        x, y_u, y_l, _, _ = self.compute_coords(inputs)

        t0 = time.time()
        cd, cm = analyze_airfoil(
            x,
            y_u,
            y_l,
            inputs["Cl_des"][0],
            inputs["Re"][0],
            inputs["M"][0],
            self.options["_xf"],
            self.options["_pool"],
            self.options["print"],
        )
        dt = time.time() - t0

        outputs["Cd"] = cd if not np.isnan(cd) else 1e27
        outputs["Cm"] = cm if not np.isnan(cm) else 1e27

        if self.options["print"]:
            print(
                f"{rank:02d} :: "
                + "a_c: {}, ".format(
                    np.array2string(
                        inputs["a_c"], separator=", ", formatter=array_formatter
                    )
                )
                + "a_t: {}, ".format(
                    np.array2string(
                        inputs["a_t"], separator=", ", formatter=array_formatter
                    )
                )
                + f't_te: {inputs["t_te"][0]: 6.4f}, '
                + f"C_d: {cd: 7.4f}, Cm: {cm: 7.4f}, dt: {dt:6.3f}"
            )


class Geom(AirfoilComponent):
    """
    Computes the thickness-over-chord ratio and cross-sectional area of an airfoil.
    """

    def setup(self):
        super().setup()
        # Outputs
        self.add_output("t_c", val=0.0)
        self.add_output("A_cs", val=0.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        x, _, _, _, t = self.compute_coords(inputs)
        outputs["t_c"] = np.max(t)
        outputs["A_cs"] = np.trapz(t, x)


class AfOptModel(om.Group):
    """
    Airfoil shape optimization using XFoil.
    """

    def initialize(self):
        self.options.declare("n_c", default=6, types=int)
        self.options.declare("n_t", default=6, types=int)
        self.options.declare("fix_te", default=True, types=bool)

        self.options.declare("constrain_thickness", default=True, types=bool)
        self.options.declare("constrain_area", default=True, types=bool)
        self.options.declare("constrain_moment", default=True, types=bool)

        self.options.declare("n_coords", default=100, types=int)

    def setup(self):
        # Number of CST coefficients
        n_c = self.options["n_c"]
        n_t = self.options["n_t"]

        # Design variable bounds
        a_c_lower = -np.ones(n_c)
        a_c_upper = np.ones(n_c)
        a_t_lower = 0.01 * np.ones(n_t)
        a_t_upper = 0.6 * np.ones(n_t)
        t_te_lower = 0.0
        t_te_upper = 0.1

        # Independent variables
        ivc = om.IndepVarComp()
        ivc.add_output("a_c", val=np.zeros(n_c))
        ivc.add_output("a_t", val=np.zeros(n_t))
        ivc.add_output("t_te", val=0.0)
        ivc.add_output("Re", val=1e6)
        ivc.add_output("M", val=0.0)
        ivc.add_output("Cl_des", val=1.0)
        ivc.add_output("Cd_0", val=1.0)
        ivc.add_output("Cm_ref", val=1.0)
        ivc.add_output("t_c_0", val=1.0)
        ivc.add_output("A_cs_0", val=1.0)

        # Main sub-systems
        self.add_subsystem("ivc", ivc, promotes=["*"])
        self.add_subsystem("XFoil", XFoilComp(n_c=n_c, n_t=n_t), promotes=["*"])

        # Design variables
        self.add_design_var("a_c", lower=a_c_lower, upper=a_c_upper)
        self.add_design_var("a_t", lower=a_t_lower, upper=a_t_upper)

        if not self.options["fix_te"]:
            self.add_design_var("t_te", lower=t_te_lower, upper=t_te_upper)

        # Objective
        self.add_subsystem(
            "F", om.ExecComp("obj = Cd / Cd_0", obj=1, Cd=1.0, Cd_0=1.0), promotes=["*"]
        )
        self.add_objective("obj")  # Cd

        # Constraints
        self.add_subsystem("Geom", Geom(n_c=n_c, n_t=n_t), promotes=["*"])

        if self.options["constrain_thickness"]:
            self.add_subsystem(
                "G1",
                om.ExecComp("g1 = 1 - t_c / t_c_0", g1=0.0, t_c=1.0, t_c_0=1.0),
                promotes=["*"],
            )
            self.add_constraint("g1", upper=0.0)  # t_c >= t_c_0

        if self.options["constrain_area"]:
            self.add_subsystem(
                "G2",
                om.ExecComp("g2 = 1 - A_cs / A_cs_0", g2=0, A_cs=1.0, A_cs_0=1.0),
                promotes=["*"],
            )
            self.add_constraint("g2", upper=0.0)  # A_cs >= A_cs_0

        if self.options["constrain_moment"]:
            self.add_subsystem(
                "G3",
                om.ExecComp(
                    "g3 = 1 - abs(Cm) / abs(Cm_ref)", g3=0.0, Cm=1.0, Cm_ref=1.0
                ),
                promotes=["*"],
            )
            self.add_constraint("g3", lower=0.0)  # |Cm| <= |Cm_max|

    def __repr__(self):
        outputs = dict(self.list_outputs(out_stream=None))
        s = ""
        s += (
            f"Con.t_c: {'True' if self.options['constrain_thickness'] else 'False'}, "
            f"Con.A_cs: {'True' if self.options['constrain_area'] else 'False'}. "
            f"Con.Cm: {'True' if self.options['constrain_moment'] else 'False'}, \n"
        )
        s += (
            f"Obj: {outputs['F.obj']['value'][0]:6.4f}, "
            f"C_l_des: {outputs['ivc.Cl_des']['value'][0]:6.4f}, "
            f"C_m_ref: {outputs['ivc.Cm_ref']['value'][0]: 7.4f}, \n"
        )
        s += (
            f"C_d: {outputs['XFoil.Cd']['value'][0]:6.4f}, "
            f"C_m: {outputs['XFoil.Cm']['value'][0]: 7.4f}, \n"
        )
        s += f"a_c: {np.array2string(outputs['ivc.a_c']['value'], formatter=array_formatter, separator=', ')}, \n"
        s += f"a_t: {np.array2string(outputs['ivc.a_t']['value'], formatter=array_formatter, separator=', ')}, \n"
        s += f"t_te: {outputs['ivc.t_te']['value'][0]: 7.4f}"
        return s


def get_ga_driver(b_c, b_t, b_te=None, gen=100, seed=None):
    # Set a starting seed for the random number generator if given
    if rank == 0:
        if seed is None:
            seed = int(SystemRandom().random() * (2 ** 31 - 1))
        print(f"SimpleGADriver_seed: {seed}")
        os.environ["SimpleGADriver_seed"] = str(seed)

    bits = {"a_c": b_c, "a_t": b_t}
    if b_te is not None:
        bits = bits.update({"t_te": b_te})

    driver = SimpleGADriver(bits=bits, run_parallel=run_parallel, max_gen=gen)
    return driver


def problem2string(prob, dt):
    """
    Return a representation of the state of the optimization problem.

    Parameters
    ----------
    prob : openmdao.api.Problem
        Airfoil optimization problem
    dt : float
        Time in seconds elapsed since last evaluation
    """
    s = prob.model.__repr__() + ",\n"
    if isinstance(prob.driver, om.SimpleGADriver):
        s += f"b_c: {prob.driver.options['bits']['a_c']}, "
        s += f"b_t: {prob.driver.options['bits']['a_t']}, "
        if not prob.model.options["fix_te"]:
            s += f"b_te: {prob.driver.options['bits']['t_te']}, \n"
    s += f"Time elapsed: {timedelta(seconds=dt)}"
    return s


def analyze(prob, initial=True, set_cm_ref=False):
    """
    Simply analyze the airfoil once.

    Parameters
    ----------
    prob : openmdao.api.Problem
        Airfoil optimization problem
    initial : bool, optional
        True if initial references values should be set based on this analysis. True by default.
    set_cm_ref : bool, optional
        True if the initial value of Cm should be used for Cm_ref. False by default.

    Returns
    -------
    openmdao.api.Problem
    """
    prob.run_model()

    if initial:
        prob["Cd_0"] = prob["Cd"]
        prob["t_c_0"] = prob["t_c"]
        prob["A_cs_0"] = prob["A_cs"]
        if set_cm_ref:
            prob["Cm_ref"] = prob["Cm"]

        prob.run_model()
    return prob


def get_coords(prob):
    """
    Get the coordinates of the airfoil represented by the current state of the airfoil optimization problem.

    Parameters
    ----------
    prob : openmdao.api.Problem
        Airfoil optimization problem

    Returns
    -------
    np.ndarray
        (n, 2) array of x-, and y-coordinates of the airfoil in counterclockwise direction
    """
    x, y_u, y_l, _, _ = cst2coords(prob["a_c"], prob["a_t"], prob["t_te"])
    x = np.reshape(x, (-1, 1))
    y_u = np.reshape(y_u, (-1, 1))
    y_l = np.reshape(y_l, (-1, 1))

    coords_u = np.concatenate((x, y_u), axis=1)
    coords_l = np.concatenate((x, y_l), axis=1)
    coords = np.concatenate((np.flip(coords_u[1:], axis=0), coords_l))

    return coords


def plot(prob, show_legend=False, show_title=True, display=False):
    """
    Plot the airfoil represented by the current state of the airfoil optimization problem.

    Parameters
    ----------
    prob : openmdao.api.Problem
        Airfoil optimization problem
    show_legend : bool, optional
        True if a legend should be shown. False by default
    show_title : bool, optional
        True if a title should be shown based on the current state of the problem. True by default
    display : bool, optional
        True if the figure should be displayed. False by default

    Returns
    -------
    figure
    """
    import matplotlib.pyplot as plt

    fig = plt.figure()
    x, y_u, y_l, y_c, _ = cst2coords(prob["a_c"], prob["a_t"], prob["t_te"])
    plt.plot(
        coords_ref[:, 0], coords_ref[:, 1], "k", x, y_u, "r", x, y_l, "r", x, y_c, "r--"
    )
    plt.axis("scaled")
    if show_legend:
        plt.legend(["Original", "Optimized"])
    if show_title:
        plt.title(prob.model)
    if display:
        fig.show()
    return fig


def write(prob, filename):
    """
    Write airfoil coordinates represented by the current state of the airfoil optimization problem to a file

    Parameters
    ----------
    prob : openmdao.api.Problem
        Airfoil optimization problem
    filename : str
        Filename
    """
    coords = get_coords(prob)
    fmt_str = 2 * ("{: >" + str(6 + 1) + "." + str(6) + "f} ") + "\n"
    with open(filename, "w") as f:
        for i in range(coords.shape[0]):
            f.write(fmt_str.format(coords[i, 0], coords[i, 1]))


def main(
    cl,
    n_c,
    n_t,
    b_c=8,
    b_t=8,
    b_te=8,
    gen=100,
    fix_te=True,
    constrain_thickness=True,
    constrain_area=True,
    constrain_moment=True,
    cm_ref=None,
    seed=None,
    repr_file="repr.txt",
    dat_file="optimized.dat",
    png_file="optimized.png",
):
    """
    Create, analyze, optimize airfoil, and write optimized coordinates to a file. Then clean the problem up and exit.

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
    repr_file, dat_file, png_file : str, optional
        Paths where the final representation, optimized airfoil coordinates, and output image should be saved.
    """
    # Construct the OpenMDAO Problem
    kwargs = dict(
        n_c=n_c,
        n_t=n_t,
        fix_te=fix_te,
        constrain_thickness=constrain_thickness,
        constrain_area=constrain_area,
        constrain_moment=constrain_moment,
        num_par_fd=n_c + n_t + int(fix_te)
    )

    prob = om.Problem()
    prob.model = AfOptModel(**kwargs)

    prob.driver = get_ga_driver(b_c, b_t, b_te if not fix_te else None, gen, seed)
    prob.setup()

    # Set the reference airfoil as initial conditions
    prob["a_c"], prob["a_t"], prob["t_te"] = coords2cst(
        x_ref, y_u_ref, y_l_ref, n_c, n_t
    )
    # Set reference values
    prob["Cl_des"] = cl
    if cm_ref is not None:
        prob["Cm_ref"] = cm_ref

    # Analyze the reference airfoil and set reference values based on initial run
    prob.run_model()

    prob["Cd_0"] = prob["Cd"]
    prob["t_c_0"] = prob["t_c"]
    prob["A_cs_0"] = prob["A_cs"]
    if cm_ref is None:
        prob["Cm_ref"] = prob["Cm"]

    # Run model one more time to have a consistent starting point
    t0 = time.time()
    prob.run_model()
    dt = time.time() - t0

    # Print results for reference airfoil
    if rank == 0:
        print("Reference airfoil:")
        print(problem2string(prob, dt))

    # Optimize the problem using a genetic algorithm
    t0 = time.time()
    prob.run_driver()
    dt = time.time() - t0

    # Run model one more time to ensure consistency
    prob.run_model()

    # Show and write final results
    if rank == 0:
        s = problem2string(prob, dt)
        print("Optimized airfoil:")
        print(s)

        with open(repr_file, "w") as f:
            f.write(s)
        write(prob, filename=dat_file)
        fig = plot(prob)
        fig.savefig(png_file)

    # Clean up and exit
    prob.cleanup()
    del prob

    sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) == 17:
        main(
            cl=float(sys.argv[1]),
            n_c=int(sys.argv[2]),
            n_t=int(sys.argv[3]),
            b_c=int(sys.argv[4]),
            b_t=int(sys.argv[5]),
            b_te=int(sys.argv[6]),
            gen=int(sys.argv[7]),
            fix_te=(sys.argv[8] == "True"),
            constrain_thickness=(sys.argv[9] == "True"),
            constrain_area=(sys.argv[10] == "True"),
            constrain_moment=(sys.argv[11] == "True"),
            cm_ref=None if sys.argv[12] == "None" else float(sys.argv[12]),
            seed=None if sys.argv[13] == "None" else int(sys.argv[13]),
            repr_file=sys.argv[14],
            dat_file=sys.argv[15],
            png_file=sys.argv[16],
        )
    else:
        main(1.0, 3, 3, constrain_moment=False, gen=9)
