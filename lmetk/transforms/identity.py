"""Identity transform that does nothing to the input data."""


class IdentityTransform():
    r"""An identity transform that does nothing to the input data.

    Parameters
    ----------
    **kwargs : optional
        Parameters for the base transform.
    """

    def __init__(self, **kwargs):
        super().__init__()
        
        # Auto-assign keyword arguments to instance attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.parameters = kwargs

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in self._init_kwargs.items())
        return f"{self.__class__.__name__}({args})"

    def forward(self, data: dict):
        r"""Apply the transform to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        dict
            The same data.
        """
        return data