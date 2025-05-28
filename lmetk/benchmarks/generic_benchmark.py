class GenericBenchmark:
    def __init__(self, cfg, objs):
        self.cfg = cfg
        self.objs = objs

    def setup(self):
        pass

    def run(self):
        pass

    def evaluate(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def __str__(self):
        return f"Experiment: {self.cfg.info.exp_name}"
