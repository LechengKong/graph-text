import torch
from tqdm import tqdm

from utils.utils import SmartTimer

class Trainer():
    def __init__(self, epoch, params):
        self.epoch = epoch
        self.optimizer = None
        self.learner = None
        self.timer = SmartTimer(False)
        self.params = params

    def full_epoch(self, learners, device=None):
        train_metric = self.train_scheduled(learners, device)
        eval_metric = self.eval_scheduled(learners, device)
        return eval_metric

    def train_scheduled(self, learners, device=None):
        train_metric = self.train_epoch(learners[0], learners[0].optimizer, device=device)
        print(train_metric)
        return train_metric
    
    def eval_scheduled(self, learners, device=None):
        eval_metric = self.eval_epoch(learners[1][0], device=device)
        print(eval_metric)
        return eval_metric

    def train_epoch(self, learner, optimizer, device=None):
        dataloader = learner.create_dataloader(dry_run=self.params.dry_run, num_workers=self.params.num_workers)
        pbar = tqdm(dataloader)
        learner.preprocess(device=device)
        self.timer.record()
        for batch in pbar:
            self.timer.cal_and_update('data')
            data = learner.load(batch, device)
            self.timer.cal_and_update('move')
            res = learner.forward_func(data)
            self.timer.cal_and_update('forward')
            loss = learner.loss_func(res, data)
            self.timer.cal_and_update('loss')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.timer.cal_and_update('back')
            with torch.no_grad():
                learner.score_func(loss, res, data)
            self.timer.cal_and_update('score')
        metrics = learner.epoch_results()
        learner.initialize_metrics()
        return metrics
    
    def eval_epoch(self, learner, device=None):
        dataloader = learner.create_dataloader(num_workers=self.params.num_workers)
        pbar = tqdm(dataloader)
        with torch.no_grad():
            learner.preprocess(device=device)
            self.timer.record()
            for batch in pbar:
                self.timer.cal_and_update('data')
                data = learner.load(batch, device)
                self.timer.cal_and_update('move')
                res = learner.forward_func(data)
                self.timer.cal_and_update('forward')
                loss = learner.loss_func(res, data)
                learner.score_func(loss, res, data)
                self.timer.cal_and_update('loss')
        metrics = learner.epoch_results()
        learner.initialize_metrics()
        return metrics

    def get_save_learner(self, learner):
        return learner[0]

    def eval_metric(self, metrics, metric_name, presult):
        return metrics[metric_name]<presult, metrics[metric_name]

    def init_metric(self):
        return 1000000

class MaxTrainer(Trainer):
    def __init__(self, epoch, params):
        super().__init__(epoch, params)
    
    def init_metric(self):
        return 0

    def eval_metric(self, metrics, metric_name, presult):
        return metrics[metric_name]>presult, metrics[metric_name]

class FilteredTrainer(Trainer):
    def __init__(self, epoch, params):
        super().__init__(epoch, params)

    def eval_scheduled(self, learners, device=None):
        eval_metric1 = self.eval_epoch(learners[1][0], device=device)
        print(eval_metric1)
        eval_metric2 = self.eval_epoch(learners[1][1], device=device)
        print(eval_metric2)
        eval_metric = {}
        for k in eval_metric1:
            eval_metric[k] = (eval_metric1[k] + eval_metric2[k])/2
        print(eval_metric)
        return eval_metric

class FilteredMaxTrainer(FilteredTrainer, MaxTrainer):
    def __init__(self, epoch, params):
        super().__init__(epoch, params)

class MultiForwardTrainer(Trainer):
    def train_scheduled(self, learners, device=None):
        cca_metric = self.train_epoch(learners[0][0], learners[0][0].optimizer, device=device)
        print(cca_metric)
        # cca_metric=None
        proj_metric = self.eval_epoch(learners[0][1], device=device)
        print(proj_metric)
        task_metric = self.train_epoch(learners[0][2], learners[0][2].optimizer, device=device)
        print(task_metric)
        return [cca_metric, task_metric]
    
    def eval_scheduled(self, learners, device=None):
        eval_metric = self.eval_epoch(learners[1][1], device=device)
        print(eval_metric)
        return eval_metric

    def get_save_learner(self, learner):
        return learner[0][0]

class RegCoTrainer(Trainer):
    def train_scheduled(self, learners, device=None):
        proj_metric = self.eval_epoch(learners[0][0], device=device)
        print(proj_metric)
        task_metric = self.train_epoch(learners[0][1], learners[0][1].optimizer, device=device)
        print(task_metric)
        return [task_metric]

    def eval_scheduled(self, learners, device=None):
        eval_metric = self.eval_epoch(learners[1][0], device=device)
        print(eval_metric)
        return eval_metric

class ProjTrainer(Trainer):
    def eval_scheduled(self, learners, device=None):
        proj_metric = self.eval_epoch(learners[1][0], device=device)
        print(proj_metric)
        eval_metric = self.eval_epoch(learners[1][1], device=device)
        print(eval_metric)
        return eval_metric