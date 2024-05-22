import torch
import logging
import os
import math

def opt_constructor(scheduler,
        nfm,
        lr,

        warm_up = None,
        fianl_step = None,
        start_lr = None,
        ref_lr = None,
        final_lr = None,
        start_wd = None,
        final_wd = None
        ):
    log_loc = os.environ.get("log_loc")
    root_dir = os.getcwd() 
    logging.basicConfig(filename=os.path.join(root_dir, f'{log_loc}/log_all'), level=logging.INFO,
                        format = '%(asctime)s - %(name)s - %(message)s')
    logger = logging.getLogger('From Optimizer')
    if not scheduler:
        lr = lr #5e-4#8e-6 
        param_groups = [
            {
                'params': (p for n, p in nfm.named_parameters()
                        if ('bias' not in n)), 
                'lr': lr,
            },{
                'params': (p for n, p in nfm.named_parameters()
                        if ('bias' in n)), 
                'lr': lr
            }
        ]
        logger.info("Optimizer is constructed")
        opt = torch.optim.Adam(param_groups) # TODO
        logger.info(f"Optimizer type: {type(opt).__name__}")
        return opt, None, None
    
    else:
        param_groups = [
            {
                'params': (p for n, p in nfm.named_parameters()
                        if ('bias' not in n) ),
                'WD_exclude': True if (start_wd == 0) and (final_wd == 0) else False,
                'weight_decay': 0
            },

            {
                'params': (p for n, p in nfm.named_parameters()
                        if ('bias' in n)),
                'WD_exclude': True if (start_wd == 0) and (final_wd == 0) else False,
                'weight_decay': 0
            }
        ]
        opt = torch.optim.AdamW(param_groups)
        scheduler = WarmupCosineSchedule(
            opt,
            warmup_steps=warm_up,
            start_lr=start_lr,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=fianl_step)
        if (start_wd == 0) and (final_wd == 0):
            wd_scheduler = None
        else:
            wd_scheduler = CosineWDSchedule(
                opt,
                ref_wd=start_wd,
                final_wd=final_wd,
                T_max=fianl_step)
            
        logger.info(f"Warm up steps: {warm_up}")
        logger.info(f"Final steps: {fianl_step}")

        logger.info("Optimizer (AdamW) with wd and lr scheduler construction was successful")
        return opt, scheduler, wd_scheduler

#############################################

class WarmupCosineSchedule(object):

    def __init__(
        self,
        optimizer,
        warmup_steps,
        start_lr,
        ref_lr,
        T_max,
        last_epoch=-1,
        final_lr=0.
    ):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps
        self._step = 0.

    def step(self):
        self._step += 1
        if self._step < self.warmup_steps:
            progress = float(self._step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            # -- consine annealing after warmup
            progress = float(self._step - self.warmup_steps) / float(max(1, self.T_max))
            new_lr = max(self.final_lr,
                         self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1. + math.cos(math.pi * progress)))

        for group in self.optimizer.param_groups:
            group['lr'] = new_lr

        return new_lr


class CosineWDSchedule(object):

    def __init__(
        self,
        optimizer,
        ref_wd,
        T_max,
        final_wd=0.
    ):
        self.optimizer = optimizer
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.T_max = T_max
        self._step = 0.

    def step(self):
        self._step += 1
        progress = self._step / self.T_max
        new_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (1. + math.cos(math.pi * progress))

        if self.final_wd <= self.ref_wd:
            new_wd = max(self.final_wd, new_wd)
        else:
            new_wd = min(self.final_wd, new_wd)

        for group in self.optimizer.param_groups:
            if ('WD_exclude' not in group) or not group['WD_exclude']:
                group['weight_decay'] = new_wd
        return new_wd