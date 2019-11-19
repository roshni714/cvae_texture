import torch

class AverageMeter(object):
    """Computes and stores the averages over a numbers or dicts of numbers.
    For the dict, this class assumes that no new keys are added during
    the computation.
    """

    def __init__(self):
        self.last_val = 0
        self.avg = 0 
        self.count = 0 

    def update(self, val, n=1):
        self.last_val = val
        n = float(n)
        if type(val) == dict:
            if self.count == 0:
                self.avg = copy.deepcopy(val)
            else:
                for key in val:
                    self.avg[key] *= self.count / (self.count + n)
                    self.avg[key] += val[key] * n / (self.count + n)
        else:
            self.avg *= self.count / (self.count + n)
            self.avg += val * n / (self.count + n)

        self.count += n
        self.last_val = val
