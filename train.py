from omegaconf import DictConfig, OmegaConf
import hydra, logging, os, itertools, glob
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from geomloss import SamplesLoss
from src.data import TriangleSampling, CombinedIterator
from src.utils import TicToc, save_checkpoint
from src.models import MongeNet

# A logger for this file
logger = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name='train')
def train_app(cfg):

    # override configuration with a user defined config file
    if cfg.user_config is not None:
        user_config = OmegaConf.load(cfg.user_config)
        cfg = OmegaConf.merge(cfg, user_config)
    logger.info('Training MongeNet\nConfig:\n{}'.format(OmegaConf.to_yaml(cfg)))
    os.makedirs(cfg.trainer.output_dir, exist_ok=True)

    # load datasets
    train_dl, test_dl = [], []
    for data_path in glob.glob(cfg.trainer.data_glob_str):        
        # read files
        triplets, ylist = sorted(glob.glob(os.path.join(data_path,'x*'))), sorted(glob.glob(os.path.join(data_path, 'y*')))          
        # create train set
        training_set = TriangleSampling(triplets[:cfg.trainer.num_train_samples], ylist[:cfg.trainer.num_train_samples])
        training_generator = DataLoader(training_set, batch_size=cfg.trainer.batch_size, shuffle=True, num_workers=0,  pin_memory=True, worker_init_fn=lambda x: np.random.seed())
        train_dl.append(training_generator)
        #create test set
        test_set = TriangleSampling(triplets[cfg.trainer.num_train_samples:cfg.trainer.num_train_samples+cfg.trainer.num_test_samples],ylist[cfg.trainer.num_train_samples:cfg.trainer.num_train_samples+cfg.trainer.num_test_samples])
        test_generator = DataLoader(test_set, batch_size=cfg.trainer.batch_size, shuffle=True, num_workers=0,  pin_memory=True, worker_init_fn=lambda x: np.random.seed())
        test_dl.append(test_generator)
    logger.info('{} training data loaders and {} test data loaders read!'.format(len(train_dl), len(test_dl)))

    # tensorboard logger
    tb_log_folder = os.path.join(cfg.trainer.output_dir, 'tb_logs')
    tb_writer = SummaryWriter(tb_log_folder)
    logger.info("Tensorboard logs in {}".format(tb_log_folder))

    # model, optimizer, and criterion
    device_used = cfg.mongenet.device
    model = MongeNet(cfg).to(device_used)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.trainer.adam_lr)  
    criterion = SamplesLoss(loss="sinkhorn", backend='tensorized', reach=None, debias=True, blur=0.00005, scaling=0.9)  
    logger.info("MongeNet Model: {}".format(model));logger.info("MongeNet Optim: {}".format(optimizer))

    # train loop
    num_of_reps, tictoc = 2, TicToc()
    for i in range(1, cfg.trainer.num_epochs+1):
        tictoc.tic('epoch')
        train_loss, train_loss_aprox, train_loss_reg = 0.0, 0.0, 0.0
        train_combined_iterator = itertools.islice(CombinedIterator(train_dl), cfg.trainer.train_epoch_size)
        Yindex, Yidxs = model.output_index, torch.unique(model.output_index)
        for X,Y in train_combined_iterator:         
            # read batch and perform predictions
            X, Y = X.float().to(device_used), Y.float().to(device_used)
            optimizer.zero_grad()
            Ypred = torch.stack([model(X)[0] for _ in range(num_of_reps)], dim=0)
            Y = torch.stack([Y for _ in range(num_of_reps)], dim=0)
            
            # approximation         
            approxvalue = 0.0
            Ypred_l = Ypred.view(num_of_reps*cfg.trainer.batch_size, model.num_outputs, 2)
            Y_l = Y.view(num_of_reps*cfg.trainer.batch_size, Y.shape[-2], 2)
            for idx in Yidxs:               
                approxvalue += criterion(Ypred_l[:, Yindex == idx], Y_l).sum()
            approxvalue = approxvalue / float(len(Yidxs)*cfg.trainer.batch_size*num_of_reps)
            train_loss_aprox += approxvalue.item()

            # diversity
            regvalue = 0.0
            for idx in Yidxs:
                regvalue += (-1. * criterion(Ypred[0][:, Yindex == idx], Ypred[1][:, Yindex == idx]).sum())
            regvalue = regvalue / float(len(Yidxs)*cfg.trainer.batch_size)
            train_loss_reg += regvalue.item()
            
            # final loss            
            lossvalue = approxvalue + cfg.trainer.reg_coef*regvalue
            train_loss += lossvalue.item()
            lossvalue.backward()
            optimizer.step()            
                         
        # log losses       
        train_loss = train_loss/float(cfg.trainer.train_epoch_size); tb_writer.add_scalar('Train/loss', train_loss, i)
        train_loss_aprox = train_loss_aprox/float(cfg.trainer.train_epoch_size); tb_writer.add_scalar('Train/approx_loss', train_loss_aprox, i)
        train_loss_reg = train_loss_reg/float(cfg.trainer.train_epoch_size); tb_writer.add_scalar('Train/diversity_loss', train_loss_reg, i)
        tb_writer.add_scalar("Train/lr", optimizer.param_groups[0]['lr'], i)
        
        # log training progress
        logger.info('Train Epoch: {} - Training Loss: {:.6f}, Approx Loss: {:.6f}, Diversity Loss: {:.6f}, LR: {:1.2e} in {:3.4f} sec'.format(
            i, train_loss, train_loss_aprox, train_loss_reg, optimizer.param_groups[0]['lr'], tictoc.toc('epoch')))
        
        # evaluate iteration
        if i % cfg.trainer.test_inteval == 0:
            tictoc.tic('test_it')
            # obs: turn on the dropout also during inference
            model.train()
            with torch.no_grad():
                # compute test loss             
                test_loss, test_loss_aprox, test_loss_reg = 0.0, 0.0, 0.0
                test_combined_iterator = itertools.islice(CombinedIterator(test_dl), cfg.trainer.test_epoch_size) 
                for X,Y in test_combined_iterator:
                    X, Y = X.float().to(device_used), Y.float().to(device_used)
                    Ypred = torch.stack([model(X)[0] for _ in range(num_of_reps)], dim=0)
                    Y = torch.stack([Y for _ in range(num_of_reps)], dim=0)                    
                    
                    # approximation         
                    approxvalue = 0.0
                    Ypred_l = Ypred.view(num_of_reps*cfg.trainer.batch_size, model.num_outputs, 2)
                    Y_l = Y.view(num_of_reps*cfg.trainer.batch_size, Y.shape[-2], 2)
                    for idx in Yidxs:               
                        approxvalue += criterion(Ypred_l[:, Yindex == idx], Y_l).sum()
                    approxvalue = approxvalue / float(len(Yidxs)*cfg.trainer.batch_size*num_of_reps)
                    test_loss_aprox += approxvalue.item()

                    # diversity
                    regvalue = 0.0
                    for idx in Yidxs:
                        regvalue += (-1.0 * criterion(Ypred[0][:, Yindex == idx], Ypred[1][:, Yindex == idx]).sum())                        
                    regvalue = regvalue / float(len(Yidxs)*cfg.trainer.batch_size)
                    test_loss_reg += regvalue.item()

                    lossvalue = approxvalue + cfg.trainer.reg_coef*regvalue
                    test_loss += lossvalue.item()

                # tensorboard logger
                test_loss = test_loss/float(cfg.trainer.test_epoch_size); tb_writer.add_scalar('Validation/loss', test_loss, i)
                test_loss_aprox = test_loss_aprox/float(cfg.trainer.test_epoch_size); tb_writer.add_scalar('Validation/approx_loss', test_loss_aprox, i)
                test_loss_reg = test_loss_reg/float(cfg.trainer.test_epoch_size); tb_writer.add_scalar('Validation/diversity_loss', test_loss_reg, i)
                                
                # Show predictions for multiple triangles
                X_cpu, Y_cpu = X.cpu().numpy(), Y[0].cpu().numpy()
                Ypred_cpu, Yindex_cpu = Ypred.cpu().detach().numpy(), Yindex.cpu().numpy()
                
                # plot predictions
                num_cols = len(Yidxs) + 1 
                fig, axis = plt.subplots(4, num_cols, figsize=(10*15,20));
                for g in range(4):
                    pts, ptsGT = Ypred_cpu[0, g], Y_cpu[g]
                    for s in range(num_cols):           
                        # axis[g, s].plot(X_cpu[g,:,0], X_cpu[g,:,1],'or');
                        axis[g, s].plot(X_cpu[g,:,0], X_cpu[g,:,1],'or'); axis[g, s].plot(X_cpu[g,[0, 1], 0], X_cpu[g, [0,1], 1], 'r');
                        axis[g, s].plot(X_cpu[g,[1, 2], 0], X_cpu[g, [1,2], 1], 'r'); axis[g, s].plot(X_cpu[g,[2, 0], 0], X_cpu[g, [2,0], 1], 'r');
                        if s == 0:
                            axis[g, s].plot(ptsGT[:, 0], ptsGT[:, 1],'.k'); 
                            axis[g, s].set_title("GT_points = {}".format(ptsGT.shape[0]))
                        else:
                            axis[g, s].plot(pts[Yindex_cpu == s, 0], pts[Yindex_cpu == s, 1],'.k');
                            axis[g, s].set_title("Pred Points = {}".format(s))
                        axis[g, s].set_ylim([0, 1.0]); axis[g, s].set_xlim([0, 1.0]);
                tb_writer.add_figure('Predictions', fig, i)     
                
                # plot diversity
                num_cols = len(Yidxs) + 1 
                fig, axis = plt.subplots(4, num_cols, figsize=(10*15, 20));
                colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']
                for g in range(4):              
                    for c_idx in range(num_cols):   
                        axis[g, c_idx].plot(X_cpu[g,:,0], X_cpu[g,:,1],'or'); axis[g, c_idx].plot(X_cpu[g,[0, 1], 0], X_cpu[g, [0,1], 1], 'r');
                        axis[g, c_idx].plot(X_cpu[g,[1, 2], 0], X_cpu[g, [1,2], 1], 'r'); axis[g, c_idx].plot(X_cpu[g,[2, 0], 0], X_cpu[g, [2,0], 1], 'r');
                        axis[g, c_idx].set_ylim([0, 1.0]); axis[g, c_idx].set_xlim([0, 1.0]);
                        for rep in range(num_of_reps):                          
                            pts, ptsGT = Ypred_cpu[rep, g], Y_cpu[g]
                            if c_idx == 0:
                                axis[g, c_idx].plot(ptsGT[:, 0], ptsGT[:, 1],'.k'); 
                                axis[g, c_idx].set_title("GT_points = {}".format(ptsGT.shape[0]))
                                break
                            else:
                                axis[g, c_idx].scatter(pts[Yindex_cpu == c_idx, 0], pts[Yindex_cpu == c_idx, 1], marker='.', c=colors[rep]);
                                axis[g, c_idx].set_title("Pred. {} points".format(c_idx))                           
                tb_writer.add_figure('Randomness', fig, i)          
            
            # save trained model at every test epoch
            save_checkpoint_file = os.path.join(cfg.trainer.output_dir, 'MongeNet_epoch{:04d}_testloss{:.4f}.tar'.format(i, test_loss))
            save_checkpoint(model, optimizer, save_checkpoint_file)
            logger.info("Checkpoint saved to {}".format(save_checkpoint_file))
            
            # print test progress            
            logger.info('Test epoch: {} - Test Loss: {:.6f}, Approx Loss: {:.6f}, Diversity Loss: {:.6f} in {:3.4f} sec (including plots and snapshoting)'.format(
                i, test_loss, test_loss_aprox, test_loss_reg, tictoc.toc('test_it')))      

    # save trained model at the final epoch
    save_checkpoint_file = os.path.join(cfg.trainer.output_dir, 'MongeNet_epoch{:04d}_testloss{:.4f}.tar'.format(i, test_loss))
    save_checkpoint(model, optimizer, save_checkpoint_file)
    logger.info("Final Checkpoint saved to {}".format(save_checkpoint_file))


if __name__ == "__main__":
    train_app()