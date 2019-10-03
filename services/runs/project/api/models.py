from project import db


class Run(db.Model):
    __tablename__ = "runs"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    run_name = db.Column(db.String(128), nullable=True)
    cl = db.Column(db.Float(), nullable=False)
    n_c = db.Column(db.Integer(), nullable=False)
    n_t = db.Column(db.Integer(), nullable=False)
    gen = db.Column(db.Integer(), nullable=False, default=100)
    tolx = db.Column(db.Float(), nullable=False)
    tolf = db.Column(db.Float(), nullable=False)
    fix_te = db.Column(db.Boolean(), nullable=False, default=True)
    t_te_min = db.Column(db.Float(), nullable=True, default=0.)
    t_c_min = db.Column(db.Float(), nullable=True, default=0.01)
    A_cs_min = db.Column(db.Float(), nullable=True, default=None)
    Cm_max = db.Column(db.Float(), nullable=True, default=None)
    strategy = db.Column(db.String(), default="rand-to-best/1/exp/random", nullable=False)
    f = db.Column(db.Float(), default=None, nullable=True)
    cr = db.Column(db.Float(), default=None, nullable=True)
    adaptivity = db.Column(db.Integer(), default=2, nullable=False)
    n_proc = db.Column(db.Integer(), nullable=False, default=28)
    report = db.Column(db.Boolean(), nullable=False, default=False)
    status = db.Column(db.Integer(), nullable=False, default=0)
    progress = db.Column(db.JSON, nullable=False, default=None)

    def __init__(
        self,
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
        cm_ref=None,
        strategy="rand-to-best/1/exp/random",
        f=None,
        cr=None,
        adaptivity=2,
        n_proc=28,
        run_name=None,
        report=False,
        progress=None,
    ):
        self.run_name = run_name
        self.cl = cl
        self.n_c = n_c
        self.n_t = n_t
        self.gen = gen
        self.tolx = tolx
        self.tolf = tolf
        self.fix_te = fix_te
        self.t_te_min = t_te_min
        self.t_c_min = t_c_min
        self.A_cs_min = A_cs_min
        self.Cm_max = Cm_max
        self.cm_ref = cm_ref
        self.strategy = strategy
        self.f = f
        self.cr = cr
        self.adaptivity = adaptivity
        self.n_proc = n_proc
        self.report = report
        self.progress = progress

    def __iter__(self):
        values = vars(self)
        for attr in self.__mapper__.columns.keys():
            if attr in values:
                yield attr, values[attr]

    def to_json(self):
        return dict([(key, value) for key, value in self.__iter__()])
