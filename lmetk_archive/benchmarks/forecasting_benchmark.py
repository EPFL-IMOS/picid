class ForecastingBenchmark:
    def __init__(self, cfg, objs, verbose=False):
        self.cfg = cfg
        self.objs = objs
        self.verbose = verbose

    def setup(self):
        pass

    def run(self):
        pass

    def evaluate(self, y, preds):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def __str__(self):
        return f"Experiment: {self.cfg.info.exp_name}"
