import time
import datetime
import logging
import torch
import sys
import numpy as np

class Trainer:
    """
    Trainer class for training a neural network model in PyTorch.


    Parameters
    ----------

    model : The neural network model to train.

    loss_fn : The loss function to use for training.

    optimizer : The optimizer to use for training.

    device : The device to use for training. Can be either 'cpu', 'mps' or 'cuda'.

    fname_save_every_epoch: filename for model to save after every epoch or None

    log_level :  The log level to use for logging. Can be one of the following:
        logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL
        
    Usage
    -----

    # create a trainer
    trainer = Trainer(model, lossFunction, optimizer, device, None, logLevel=logging.INFO)
    # train the model
    trainer.train(train_loader, val_loader, numberOfEpochs)

    Author
    ------
    Markus Enzweiler (markus.enzweiler@hs-esslingen.de)

    """

    def __init__(self, model, loss_fn, optimizer, device, fname_save_every_epoch=None, log_level=logging.INFO):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer  
        self.device = device
        self.fname_save_every_epoch = fname_save_every_epoch
        self.have_val_data = False
        self.train_batch_size = 0
        self.val_batch_size = 0

        # logging
        self.log_level = log_level
        self.logger = None
        self.logger_stream_handler = None
        self._setup_logger()
        
        # metrics, computed in each epoch
        self.metrics = dict()
        self.metrics["epochTrainLoss"] = []
        self.metrics["epochValLoss"] = []
        self.metrics["epochTrainAccuracy"] = []
        self.metrics["epochValAccuracy"] = []

        # timing
        self.metrics["epochStartTime"] = None
        self.metrics["epochEndTime"] = None
        self.metrics["trainingStartTime"] = None
        self.metrics["trainingEndTime"] = None
        self.metrics["throughput"] = 0.0


    def _setup_logger(self):
        logging.basicConfig(stream=sys.stdout, level = self.log_level, force=True)
        self.logger = logging.getLogger('Trainer')

        self.logger_stream_handler = logging.StreamHandler()
        self.logger_stream_handler.setLevel(self.log_level)
        formatter = logging.Formatter('%(message)s')
        self.logger_stream_handler.setFormatter(formatter)
        
        self.logger.handlers.clear()
        self.logger.addHandler(self.logger_stream_handler)
        self.logger.propagate = False

    def _save_model(self, name, epoch):
        fname = f"{name}_{epoch:03d}.pth"
        self.model.save(fname)


    def _init_metrics(self, num_epochs):
        self.metrics["epochTrainLoss"    ] = [0.0] * num_epochs
        self.metrics["epochValLoss"     ] = [0.0] * num_epochs
        self.metrics["epochTrainAccuracy"] = [0.0] * num_epochs
        self.metrics["epochValAccuracy" ] = [0.0] * num_epochs


    def _update_metrics(self, epoch, num_train_samples, num_val_samples):
        # average loss and accuracy
        if num_train_samples:
            self.metrics["epochTrainLoss"][epoch]     = self.metrics["epochTrainLoss"][epoch]     / num_train_samples
            self.metrics["epochTrainAccuracy"][epoch] = self.metrics["epochTrainAccuracy"][epoch] / num_train_samples 

        if self.have_val_data and num_val_samples:      
            self.metrics["epochValLoss"][epoch]      = self.metrics["epochValLoss"][epoch]      / num_val_samples
            self.metrics["epochValAccuracy"][epoch]  = self.metrics["epochValAccuracy"][epoch]  / num_val_samples


    def _log_metrics(self, epoch):
        # log metrics
        self.logger_stream_handler.terminator = ""
        time_taken = self.metrics["epochEndTime"] - self.metrics["epochStartTime"]
        train_loss = self.metrics["epochTrainLoss"][epoch]
        train_accuracy = self.metrics["epochTrainAccuracy"][epoch]
        throughput = self.metrics["throughput"]

        log_message = (
            f"[Epoch {epoch:3d}] : | "
            f"time: {time_taken:6.3f}s | "
            f"trainLoss: {train_loss:6.3f} | "
            f"trainAccuracy: {train_accuracy:6.3f} | "
        )

        if self.have_val_data:
            val_loss = self.metrics["epochValLoss"][epoch]
            val_accuracy = self.metrics["epochValAccuracy"][epoch]
            log_message += (
                f"valLoss: {val_loss:6.3f} | "
                f"valAccuracy: {val_accuracy:6.3f} | "
            )
        else:
            log_message += "no validation data | "

        log_message += f"throughput: {throughput:10.3f} img/s |"

        self.logger.info(log_message)
        self.logger.info('\n')


    def _on_train_begin(self, num_epochs):
        # Push the network model to the device we are using to train
        self.model.to(self.device)

        # init the metrics
        self._init_metrics(num_epochs)

        # start time
        self.metrics["trainingStartTime"] = time.monotonic()


    def _on_train_end(self, num_epochs):        
        # end time
        self.metrics["trainingEndTime"] = time.monotonic()
        timeDelta = datetime.timedelta(seconds=(self.metrics["trainingEndTime"] - self.metrics["trainingStartTime"]))

        # log
        self.logger_stream_handler.terminator = "\n"
        self.logger.info(f'Training finished in {str(timeDelta)} hh:mm:ss.ms')

    
    def _on_epoch_begin(self, epoch):
        # start time of epoch
        self.metrics["epochStartTime"] = time.monotonic()  

        # log info
        self.logger_stream_handler.terminator = " "
        self.logger.info( f'[Epoch {epoch:3}] : ') 


    def _on_epoch_end(self, epoch, num_train_samples, num_val_samples, num_batches):        
        # end time of epoch
        self.metrics["epochEndTime"] = time.monotonic()  

        # log info
        self.logger_stream_handler.terminator = "\n"
        self.logger.info(f' done ({num_batches} batches)') 

        # update metrics
        self._update_metrics(epoch, num_train_samples, num_val_samples)

        # log metrics
        self._log_metrics(epoch)

        # save model
        if self.fname_save_every_epoch:
            self._save_model(self.fname_save_every_epoch, epoch)


    def _compute_accuracy(self, outputs, labels):
         # find predicted labels (the output neuron index with the highest output value)
        _, predicted_labels = torch.max(outputs, 1) 
        return torch.sum(predicted_labels == labels).detach().cpu().numpy()
        

    def _train_epoch(self, epoch, train_loader):
         # loop over batches in the dataset
        num_batches = 0

        # throughput (images per second)
        self.metrics["throughput"] = []

        # model in training mode
        self.model.train()


        for i, data in enumerate(train_loader, 0):        
            # get the training data : data is a list of [images, labels]
            # and push the data to the device we are using       
            images, labels = data[0].to(self.device), data[1].to(self.device)

            # zero the parameter gradients before the next data batch is processed
            self.optimizer.zero_grad()

            # start time of batch
            batch_start_time = time.monotonic()  

            # forward pass of the batch
            outputs = self.model(images)

            # loss computation at the output of the network
            loss = self.loss_fn(outputs, labels)

            # backpropagate the loss through the network
            loss.backward()

            # optimize the network parameters
            self.optimizer.step()

             # end time of batch
            batch_end_time = time.monotonic()  

            # accumulate train loss
            self.metrics["epochTrainLoss"][epoch] += loss.item() * self.train_batch_size

            # compute and accumulate train accuracy  
            batch_accuracy = self._compute_accuracy(outputs, labels)             
            self.metrics["epochTrainAccuracy"][epoch] += batch_accuracy

            # throughput (images per second)
            self.metrics["throughput"].append(images.shape[0] / (batch_end_time - batch_start_time))

            if ((i % 100) == 0):
                self.logger_stream_handler.terminator = ""
                self.logger.info ('.')

            num_batches = num_batches + 1

        # average throughput over batches (images per second)
        self.metrics["throughput"] = np.mean(self.metrics["throughput"])

        return num_batches


    def _test_epoch(self, epoch, val_loader):

        # test on the validation data
        if val_loader:        

            # model in eval mode
            self.model.eval()

            # we do not compute gradients in inference mode     
            with torch.no_grad(): 

                # loop over validation data
                for data in val_loader:  

                    # push to device
                    images, labels = data[0].to(self.device), data[1].to(self.device)  

                    # forward pass through the network
                    outputs = self.model(images)

                    # compute loss (just to report it)
                    loss = self.loss_fn(outputs, labels)          
                    self.metrics["epochValLoss"][epoch] += loss.item() * self.val_batch_size

                    # compute accuracy                
                    self.metrics["epochValAccuracy"][epoch] += self._compute_accuracy(outputs, labels)


    def train(self, train_loader, val_loader, num_epochs):
        # main training method    

        # check training and validation data
        if not (train_loader and len(train_loader) > 0):
            msg = 'No training data available'
            self.logger.error(msg)
            raise Exception(msg)

        if not (val_loader and len(val_loader) > 0):
            self.have_val_data = False
        else:
            self.have_val_data = True

        # number of train and validation samples
        num_train_samples = (len(train_loader.dataset) if train_loader else 0)
        num_val_samples   = (len(val_loader.dataset)  if val_loader  else 0)

        # batch sizes of train and validation loader
        self.train_batch_size = (train_loader.batch_size if train_loader else 0)
        self.val_batch_size   = (val_loader.batch_size  if val_loader  else 0)


        # ------ Main training loop ------

        # do some stuff at the beginning of the training
        self._on_train_begin(num_epochs)

        # loop over the dataset in each epoch
        for epoch in range(num_epochs):           

            # do some stuff at the beginning of each epoch
            self._on_epoch_begin(epoch)

            # train an epoch on the training data
            num_batches = self._train_epoch(epoch, train_loader)

            # test on the validation data
            if self.have_val_data:
                self._test_epoch(epoch, val_loader)

            # do some stuff at the end of each epoch
            self._on_epoch_end(epoch, num_train_samples, num_val_samples, num_batches)

         # do some stuff at the end of the training
        self._on_train_end(num_epochs)

