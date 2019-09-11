from sqlalchemy.sql import func

from project import db


class Run(db.Model):
    __tablename__ = "runs"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    run_name = db.Column(db.String(128), nullable=True)
    cl = db.Column(db.Float(), nullable=False)
    n_c = db.Column(db.Integer(), nullable=False)
    n_t = db.Column(db.Integer(), nullable=False)
    b_c = db.Column(db.Integer(), nullable=False, default=8)
    b_t = db.Column(db.Integer(), nullable=False, default=8)
    b_te = db.Column(db.Integer(), nullable=True, default=8)
    gen = db.Column(db.Integer(), nullable=False, default=100)
    fix_te = db.Column(db.Boolean(), nullable=False, default=True)
    constrain_thickness = db.Column(db.Boolean(), nullable=False, default=True)
    constrain_area = db.Column(db.Boolean(), nullable=False, default=True)
    constrain_moment = db.Column(db.Boolean(), nullable=False, default=False)
    cm_ref = db.Column(db.Float(), nullable=True)
    seed = db.Column(db.Integer(), nullable=True)
    n_proc = db.Column(db.Integer(), nullable=False, default=16)
    report = db.Column(db.Boolean(), nullable=False, default=False)
    status = db.Column(db.Integer(), nullable=False, default=0)

    def __init__(
        self,
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
        constrain_moment=False,
        cm_ref=None,
        seed=None,
        n_proc=16,
        run_name=None,
        report=False,
    ):
        self.run_name = run_name
        self.cl = cl
        self.n_c = n_c
        self.n_t = n_t
        self.b_c = b_c
        self.b_t = b_t
        self.b_te = b_te
        self.gen = gen
        self.fix_te = fix_te
        self.constrain_thickness = constrain_thickness
        self.constrain_area = constrain_area
        self.constrain_moment = constrain_moment
        self.cm_ref = cm_ref
        self.seed = seed
        self.n_proc = n_proc
        self.report = report

    def __iter__(self):
        values = vars(self)
        for attr in self.__mapper__.columns.keys():
            if attr in values:
                yield attr, values[attr]

    def to_json(self):
        return dict([(key, value) for key, value in self.__iter__()])
