import torch
from options  import train_opts
from utils    import logger, recorders
from datasets import custom_data_loader
from models   import custom_model, solver_utils, model_utils

import train_utils
import test_utils

args = train_opts.TrainOpts().parse()
log  = logger.Logger(args)

def main(args):
    train_loader, val_loader = custom_data_loader.customDataloader(args)

    model = custom_model.buildModel(args)
    optimizer, scheduler, records = solver_utils.configOptimizer(args, model)
    criterion = solver_utils.Criterion(args)
    recorder  = recorders.Records(args.log_dir, records)

    for epoch in range(args.start_epoch, args.epochs+1):
        scheduler.step()
        recorder.insertRecord('train', 'lr', epoch, scheduler.get_lr()[0])

        train_utils.train(args, train_loader, model, criterion, optimizer, log, epoch, recorder)
        if epoch % args.save_intv == 0: 
            model_utils.saveCheckpoint(args.cp_dir, epoch, model, optimizer, recorder.records, args)

        if epoch % args.val_intv == 0:
            test_utils.test(args, 'val', val_loader, model, log, epoch, recorder)

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    main(args)
