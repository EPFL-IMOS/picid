
interface for tabularizer


class Tabularizer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def transform(self, data):
        # Implement the transformation logic here
        return data