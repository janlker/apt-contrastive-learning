import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import  StepLR, ExponentialLR, MultiStepLR
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import build_dataset
from utils.metrics import get_metrics
from models import DGCNN_cls
from omegaconf import OmegaConf

from datetime import datetime

def train(config):

    config.num_classes = len(config.phase_dict)
    g = torch.Generator()
    g.manual_seed(config.seed)
    print(config.dataset_mode + " selected")
    print("Train rotation: " + config.rotation)

    # Build dataset. see build_dataset function for assumptions made regarding partition/supervision
    train_apt_dataset = build_dataset(config=config,partition="train",supervision="supervised")
    train_loader = DataLoader(train_apt_dataset,
        num_workers=4,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        generator=g,
    )

    test_apt_dataset = build_dataset(config=config,partition="test",supervision="supervised")
    test_loader = DataLoader(test_apt_dataset,
        num_workers=4,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        generator=g,
    )

    # Define device, model, loss, optimzer
    input_dims = train_apt_dataset.dpos.shape[1] - 1 # -1 due to label column
    print("Shape of one data batch is: " + str(input_dims))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_base = DGCNN_cls(config=config,input_dims=input_dims,device=device)

    if config.num_points > 2000: 
        model = torch.compile(model_base)
    else:
        model = model_base

    print(str(model))
    model.to(device)

    # Loss and metrics
    criterion = nn.CrossEntropyLoss()

    metrics = ["f1_score", "f1_score", "ari"]
    reduction_method = ["micro", "macro", ""] # same length as metrics
    metrics_func = get_metrics(metrics=metrics, num_classes=config.num_classes, average=reduction_method, device=device)

    if config.optimizer == "adam":
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-5)
    elif config.optimizer == "adamW":
        print("Use AdamW")
        opt = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    else:
        print("Please select valid optimizer")

    if config.scheduler == "exp":
        scheduler = ExponentialLR(opt, gamma=config.decay_rate)
    elif config.scheduler == "step":
        scheduler = StepLR(opt, step_size=config.decay_step, gamma=config.decay_rate)
    elif config.scheduler == "multi_step":
        scheduler = MultiStepLR(opt, milestones=[50,70,90], gamma=config.decay_rate)
    print(scheduler)

    # Begin training
    best_test_loss = 1e10
    training_history = []
    print("Start Training", flush=True)

    for epoch in range(config.epochs):
        ####################
        # Train
        ####################
        model.train()
        train_loss_buf = []
        train_label_buf = []

        start_time = datetime.now()

        for batch_idx, (data, label) in enumerate(train_loader):
            data, label = (data.to(device, dtype=torch.float),label.to(device, dtype=torch.long).squeeze())  
            data = data.permute(0, 2, 1)
            opt.zero_grad()

            cls_output, shape_feat = model(data)
            loss = criterion(cls_output, label)

            loss.backward()
            opt.step()

            metrics_func.calculate(cls_output, label)
            train_loss_buf.append(loss.detach().cpu().numpy())
            train_label_buf.append(label.cpu())

        scheduler.step()
        if opt.param_groups[0]["lr"] < 0.000001:
            for param_group in opt.param_groups:
                param_group["lr"] = 0.000001

        train_metrics = metrics_func.compute()
        metrics_func.reset()
        train_loss_buf = np.mean(train_loss_buf)
        train_label_buf = torch.cat(train_label_buf).numpy()


        ####################
        # Test
        ####################
        test_loss_buf = []
        test_label_buf = []
        test_cls_buf = []

        model.eval()
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(test_loader):
                data, label = (data.to(device, dtype=torch.float),label.to(device, dtype=torch.long).squeeze(),)
                data = data.permute(0, 2, 1)

                cls_output, shape_feat = model(data)
                loss = criterion(cls_output, label)

                metrics_func.calculate(cls_output, label)

                test_loss_buf.append(loss.cpu().numpy())
                test_label_buf.append(label.cpu())
                test_cls_buf.append(cls_output.cpu())

        test_loss_buf = np.mean(test_loss_buf)
        test_label_buf = torch.cat(test_label_buf)
        test_cls_buf = torch.cat(test_cls_buf)
        test_metrics = metrics_func.compute()   
        metrics_func.reset()

        if test_loss_buf <= best_test_loss:
            best_test_loss = test_loss_buf
            torch.save(model_base.state_dict(),config.log_path  + "model_checkpoint.pt")  

        if epoch == config.epochs - 1:
            torch.save(model_base.state_dict(),config.log_path + "model_checkpoint_last_epoch.pt")  
        print("Epoch completed in: " + str(datetime.now()- start_time), flush=True)

        f1_train = train_metrics["f1_score_micro"]
        f1_test = test_metrics["f1_score_micro"]
        f1_train_macro = train_metrics["f1_score_macro"]
        f1_test_macro = test_metrics["f1_score_macro"]
        ari_train = train_metrics["ari"]
        ari_test = test_metrics["ari"]

        lr = opt.param_groups[0]["lr"]

        # save as txt file 
        training_history.append(
            [
                epoch,
                train_loss_buf,
                test_loss_buf,
                f1_train,
                f1_test,
                f1_train_macro,
                f1_test_macro,
                ari_train,
                ari_test,
                lr
            ]
        )
        np.savetxt(
            config.log_path  + "training_history.csv",
            np.asarray(training_history),
            delimiter=",",
            header="epoch,train_loss,test_loss,f1_train,f1_test,f1_train_macro,f1_test_macro,ari_train,ari_test,lr",

            fmt=(
                "%d",
                "%1.5f",
                "%1.5f",
                "%1.5f",
                "%1.5f",
                "%1.5f",
                "%1.5f",
                "%1.5f",
                "%1.5f",
                "%1.7f"

            ),
        )


if __name__ == "__main__":

    yaml_path = "configs/supervised_config.yaml"
    config = OmegaConf.load(yaml_path)

    config.phase_dict = config.phase_dicts[config.phase_identifier]
    config.num_classes = len(config.phase_dict)

    if not os.path.exists(config.log_path + "/run_" + str(config.run_id)): # create main folder of run
        os.makedirs(config.log_path + "/run_" + str(config.run_id))
    
    config.log_path = f"{config.log_path}/run_{config.run_id}/"
    OmegaConf.save(config=config, f=config.log_path+yaml_path.split("/")[-1])

    print("Run nbr: " + str(config.run_id))

    cuda = torch.cuda.is_available()
    torch.manual_seed(config.seed)
    if cuda:
        print(
            "Using GPU : "
            + str(torch.cuda.current_device())
            + " from "
            + str(torch.cuda.device_count())
            + " devices"
        )
        torch.cuda.manual_seed(config.seed)
    else:
        print("Using CPU")

    train(config)


