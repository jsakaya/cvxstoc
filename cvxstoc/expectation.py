import copy, queue
import numpy

from cvxpy.transforms.partial_optimize import PartialProblem
from utils import replace_rand_vars
from random_variable import RandomVariable

def construct_saa(expr, num_samples, num_burnin_samples):
    # Get samples of each RandomVariable in expr.
    rvs2samples, ignore = clamp_or_sample_rvs(
        expr,
        rvs2samples={},
        want_de=False,
        want_mf=False,
        num_samples=num_samples,
        num_burnin_samples=num_burnin_samples,
    )
    
    # If expr contained no RandomVariables, we're done.
    if not rvs2samples.keys():
        return expr

    # Else expr contained RandomVariables.
    else:
        mul_exprs = []
        for s in range(num_samples - num_burnin_samples):

            expr_copy = replace_rand_vars(expr)  # Deep copy expr (num_samples times)
            expr_copy_realized = clamp_or_sample_rvs(
                expr_copy, rvs2samples, want_de=False, want_mf=False, sample_idx=s
            )
            # Plug in a realization of each RandomVariable
            prob = 1.0 / (num_samples - num_burnin_samples)
            mul_exprs.append(expr_copy_realized)

        # Return an AddExpression using all the realized deep copies to the caller.
        return sum(mul_exprs), mul_exprs


def clamp_or_sample_rvs(
    expr,
    rvs2samples={},
    want_de=None,
    want_mf=None,
    num_samples=None,
    num_burnin_samples=None,
    sample_idx=None,
    sample_idxes=None,
):
    # Walk expr and "process" each RandomVariable.
    if not rvs2samples.keys():
        draw_samples = True
    else:
        draw_samples = False

    my_queue = queue.Queue()
    my_queue.put(ExpectationqueueItem(expr))

    rvs = []
    rv_ctr = 0
    while True:
        queue_item = my_queue.get()
        cur_expr = queue_item._item
        if isinstance(cur_expr, RandomVariable):

            if draw_samples:
                rvs += [cur_expr]

                if cur_expr not in rvs2samples:
                    samples = cur_expr.sample(num_samples, num_burnin_samples)
                    rvs2samples[cur_expr] = samples

            else:
                if want_de:
                    idx = int(sample_idxes[rv_ctr])
                    cur_expr.value = cur_expr._metadata["vals"][idx]
                    rv_ctr += 1
                elif want_mf:
                    cur_expr.value = cur_expr.mean
                else:                                        
                    cur_expr.value = rvs2samples[cur_expr][sample_idx]                                        

        else:
            if isinstance(cur_expr, PartialProblem):
                my_queue.put(ExpectationqueueItem(cur_expr.args[0].objective))

                for constr in cur_expr.args[0].constraints:
                    my_queue.put(ExpectationqueueItem(constr))

            else:
                for arg in cur_expr.args:
                    my_queue.put(ExpectationqueueItem(arg))

        if my_queue.empty():
            break

    if draw_samples:
        return rvs2samples, rvs
    else:
        return expr


class ExpectationqueueItem:
    def __init__(self, item):
        self._item = item