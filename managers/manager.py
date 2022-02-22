import torch

class Manager():
    def __init__(self, learner, trainer, save_path='model'):
        self.learner = learner
        self.trainer = trainer
        self.save_path = save_path
        self.starting_epoch = 0

    def setup(self, load_model=None):
        #setup optimizer
        #load model 
        if load_model is not None:
            #read state dict
            self.starting_epoch = self.learner[0].load_model(load_model)
            
    def train(self, trainer,metric_name= 'mrr',save_epoch=1, save_model=True, device=None):
        print('Train: Optimize w.r.t', metric_name)
        best_res = trainer.init_metric()
        save_learner = trainer.get_save_learner(self.learner)
        for epoch in range(self.starting_epoch+1, self.starting_epoch+trainer.epoch+1):
            print('Epoch', epoch)
            metrics = trainer.full_epoch(self.learner, device=device)
            update, res = trainer.eval_metric(metrics, metric_name, best_res)
            if update:
                print('Found better model')
                best_res = res
                if save_model:
                    save_learner.save_model(self.save_path+'_best.pth', epoch)
            if epoch%save_epoch==0 and save_model:
                save_learner.save_model(self.save_path+'.pth', epoch)
            #get results
            #save model

    def eval(self, learner, trainer, device=None):
        print('Eval:')
        metrics = trainer.eval_scheduled(learner, device=device)

