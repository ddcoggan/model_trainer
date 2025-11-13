
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset_epoch()
        self.reset_batches()

    def reset_epoch(self):
        self.val = 0
        self.avg_epoch = 0
        self.sum_epoch = 0
        self.count_epoch = 0

    def reset_batches(self):
        self.val = 0
        self.avg_batches = 0
        self.sum_batches = 0
        self.count_batches = 0

    def update(self, val):

        self.val = val

        self.sum_epoch += val
        self.count_epoch += 1
        self.avg_epoch = self.sum_epoch / self.count_epoch

        self.sum_batches += val
        self.count_batches += 1
        self.avg_batches = self.sum_batches / self.count_batches
