import copy

import numpy as np
import scipy

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax import random
from cvxpy import utilities, interface
from cvxpy.expressions.constants.constant import Constant
from cvxpy.expressions.constants.parameter import Parameter

class RandomVariable(Parameter):
    def __init__(self, rv=None, model=None, name=None, val_map=None, metadata=None):
        if name is not None:
            self._name = name

        self._metadata = metadata
        self._val_map = val_map
        self.set_rv_model_and_maybe_name(rv, model)
        self.set_shape()
        super(RandomVariable, self).__init__(self._shape, self._name)


    @property
    def mean(self):
        return self._metadata["mu"]

    def set_rv_model_and_maybe_name(self, rv, model):
        if rv is not None and model is None:
            self._rv = rv
            self._model = self.create_numpyro_model(rv)
        elif rv is not None and model is not None:
            self._rv = rv
            self._model = model
        elif rv is None and model is not None:
            self._model = model
            self._rv = None
            for name, dist in self._model().items():
                if name == self._name:
                    self._rv = dist
                    break
            if self._rv is None:
                raise Exception("CANT_FIND_NUMPYRO_RV_IN_NUMPYRO_MODEL_OBJ")
        else:
            raise Exception("DIDNT_PASS_EITHER_RV_OR_MODEL")

    def create_numpyro_model(self, rv):
        def model():
            numpyro.sample(self._name, rv)
        return model

    def set_shape(self):
        shape = ()
        if self.has_val_map():
            val = list(self._val_map.values())[0]
            if isinstance(val, int) or isinstance(val, float):
                shape = ()
            elif isinstance(val, np.ndarray):
                numpy_shape = val.shape
                if len(numpy_shape) == 1:
                    shape = (numpy_shape[0],)
                elif len(numpy_shape) == 2:
                    shape = (numpy_shape[0], numpy_shape[1])
                else:
                    raise Exception("BAD_RV_DIMS")
            else:
                raise Exception("BAD_VAL_MAP")
        else:
            if hasattr(self._rv, 'shape'):
                numpyro_shape = self._rv.shape()
            else:
                numpyro_shape = ()
            if len(numpyro_shape) == 0:
                shape = ()
            elif len(numpyro_shape) == 1:
                shape = (numpyro_shape[0],)
            elif len(numpyro_shape) == 2:
                shape = (numpyro_shape[0], numpyro_shape[1])
            else:
                raise Exception("BAD_RV_DIMS")
        self._shape = shape

    def name(self):
        # Override.
        if self.value is None:
            return self._name
        else:
            return str(self.value)

    def __repr__(self):
        # Override.
        return "RandomVariable(%s, %s, %s)" % (self.curvature, self.sign, self.size)

    def __eq__(self, rv):
        # Override.
        return self._name == rv._name

    def __hash__(self):
        # Override.
        return hash(self._name)

    def __deepcopy__(self, memo):
        # Override.
        return self.__class__(
            rv=self._rv,
            model=self._model,
            name=self.name(),
            val_map=self._val_map,
            metadata=self._metadata,
        )

    def sample(self, num_samples, num_burnin_samples=0):
        if num_samples == 0:
            return np.array([None])

        nuts_kernel = NUTS(self._model)
        mcmc = MCMC(nuts_kernel, num_warmup=num_burnin_samples, num_samples=num_samples)
        rng_key = random.PRNGKey(1)
        mcmc.run(rng_key)
        samples = mcmc.get_samples()[self._name]

        if not self.has_val_map():
            return np.array(samples)
        else:
            samples_mapped = [self._val_map[sample[0]] for sample in samples]
            return np.array(samples_mapped)

    def has_val_map(self):
        if self._val_map is not None and len(self._val_map.values()) > 0:
            return True
        else:
            return False

