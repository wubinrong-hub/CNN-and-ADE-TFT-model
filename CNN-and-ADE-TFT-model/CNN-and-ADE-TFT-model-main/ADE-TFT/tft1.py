import sys
sys.path.append("C:/Users/Administrator/Desktop/Desktop/ADE-TFT")
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")  # avoid printing out absolute paths
os.chdir("../../..")
import copy
from pathlib import Path
import warnings
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
import tensorflow as tf
import tensorboard as tb
def MyTFT(vdim,XXXX,bound):

 csv_path = 'C:/Users/Administrator/Desktop/ADE-TFT/data1.csv'
 data = pd.read_csv(csv_path)
 tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
 data["id"] = "tourism volume"
 data["time_idx"] = pd.to_datetime(data["data"]).astype(np.int64)
 data["time_idx"] -= data["time_idx"].min()
 data["time_idx"] = (data.time_idx / 3600000000000) + 1
 data["time_idx"] = data["time_idx"].astype(int)
 data1=data
 data = data1[:41]
 max_prediction_length = 1
 max_encoder_length = int(XXXX[0])
 training_cutoff = data["time_idx"].max() - max_prediction_length
 test_data = data[lambda x: x.time_idx > x.time_idx.max() - max_encoder_length]
 training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="tourism volume",
    group_ids=["id"],
    min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["id"],
    time_varying_known_reals=["time_idx","Month","Year"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=["tourism volume","Confirmed cases","number of posts","CNN values","Paris museum","Paris airport","Paris flight"
                                ],
    target_normalizer=GroupNormalizer(
        groups=["id"], transformation="softplus"
    ),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
 )
 validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)
 batch_size = int(XXXX[1])  # set this between 32 to 128
 train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
 val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)
 pl.seed_everything(42)
 early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=20, verbose=False, mode="min")
 lr_logger = LearningRateMonitor()  # log the learning rate
 logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard
 trainer = pl.Trainer(
    max_epochs=10,#30
    gpus=0,
    weights_summary="top",
    gradient_clip_val=0.1,
    limit_train_batches=30,  # coment in for training, running valiation every 30 batches
    callbacks=[lr_logger, early_stop_callback],
    logger=logger,
 )
 tft = TemporalFusionTransformer.from_dataset(
    training,
    #learning_rate= 0.1,
    learning_rate= XXXX[4],
    hidden_size=int(XXXX[2]),
    attention_head_size=int(XXXX[5]),
    dropout=0.1,
    hidden_continuous_size=int(XXXX[3]),
    output_size=7,  # 7 quantiles by default
    loss=QuantileLoss(),
    log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
    reduce_on_plateau_patience=4,
 )
 trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
 )
 raw_predictions, x = tft.predict(val_dataloader, mode="raw", return_x=True)
 tft.plot_prediction(x, raw_predictions, idx=0, add_loss_to_title=True)
 new_raw_predictions, new_x =  tft.predict(
    test_data,
    mode="raw",
    return_x=True
 )
 tft.plot_prediction(
    new_x,
    new_raw_predictions,
    idx=0,
    show_future_observed=False
 )
 result=[]
 len_data=len(data1)
 test_len=6
 test_y=[]
 for i in range(test_len):
    test_data=data1[len_data-max_encoder_length-(test_len-i)+1:len_data-(test_len-i)+1]
    new_raw_predictions, new_x =  tft.predict(
    test_data,
    mode="raw",
    return_x=True
    )
    a=new_raw_predictions['prediction']
    b=a.numpy()
    result.append(b[0,0,3])
 test_y1=data1["tourism volume"]
 test_y2=test_y1[len_data-test_len:len_data]
 test_y=[]
 for i in range(test_len):
    test_y.append(test_y2[[len_data+i-test_len]])
 result=np.array(result)
 test_y=np.array(test_y)
 MAPE1 = 0
 for i in range(test_len):
    MAPE1=MAPE1+np.abs(result[i]-test_y[i])/test_y[i]
 MAPE1 = MAPE1/test_len
 print(XXXX)
 return  MAPE1