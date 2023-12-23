import time
import datetime
import logging
import torch
import numpy as np

class Tester:
    """
    Tester class for testing a neural network model in PyTorch.


    Parameters
    ----------

    model : The neural network model to test.

    device : The device to use for testing. Can be either 'cpu', 'mps' or 'cuda'.

    log_level :  The log level to use for logging. Can be one of the following:
        logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL
        
    Usage
    -----

    # create a tester
    tester = Tester(model, device, log_level=logging.INFO)
    # test the model
    tester.test(test_loader)

    Author
    ------
    Markus Enzweiler (markus.enzweiler@hs-esslingen.de)

    """

    def __init__(self, model, device, log_level=logging.INFO):
        self.model = model     
        self.device = device      
        self.test_batch_size = 0

        # logging
        self.log_level = log_level
        self.logger = None
        self.logger_stream_handler = None
        self._setup_logger()
        
        # metrics
        self.metrics = dict()
        self.metrics["accuracy"] = 0.0


    def _setup_logger(self):
        logging.basicConfig(level = self.log_level, force=True)
        self.logger = logging.getLogger('Tester')

        self.logger_stream_handler = logging.StreamHandler()
        self.logger_stream_handler.setLevel(self.log_level)
        formatter = logging.Formatter('%(message)s')
        self.logger_stream_handler.setFormatter(formatter)
        
        self.logger.handlers.clear()
        self.logger.addHandler(self.logger_stream_handler)
        self.logger.propagate = False


    def _init_metrics(self):
        self.metrics["accuracy"] = 0.0


    def _update_metrics(self, num_test_samples):
        # average accuracy
        if num_test_samples:
            self.metrics["accuracy"] = self.metrics["accuracy"] / num_test_samples


    def _log_metrics(self, num_test_samples):       
        # log metrics
        self.logger_stream_handler.terminator = ""     
       
        self.logger.info('\n')
        self.logger.info(f'Test Metrics ({num_test_samples} test samples):\n')
        self.logger.info(f'  - Accuracy: {self.metrics["accuracy"]:.3f}')
        self.logger.info('\n')
           

    def _on_test_begin(self):
        # Push the network model to the device we are using to train
        self.model.to(self.device)

        # init the metrics
        self._init_metrics()

        # start time
        self.metrics["testingStartTime"] = time.monotonic()

        # logging output
        self.logger_stream_handler.terminator = ""          
        self.logger.info(f'Testing ')


    def _on_test_end(self, num_test_samples):  

        # update Metrics
        self._update_metrics(num_test_samples)

         # log metrics
        self._log_metrics(num_test_samples)

        # end time
        self.metrics["testingEndTime"] = time.monotonic()
        timeDelta = datetime.timedelta(seconds=(self.metrics["testingEndTime"] - self.metrics["testingStartTime"]))

        # log
        self.logger_stream_handler.terminator = "\n"
        self.logger.info(f'Testing finished in {str(timeDelta)} hh:mm:ss.ms')


    def _compute_accuracy(self, outputs, labels):
         # find predicted labels (the output neuron index with the highest output value)
        _, predicted_labels = torch.max(outputs, 1) 
        return torch.sum(predicted_labels == labels).detach().cpu().numpy()
        

    def test(self, test_loader):   
        # number of test samples
        num_test_samples = (len(test_loader.dataset) if test_loader else 0)
        
        # batch size of test loader
        self.test_batch_size = (test_loader.batch_size if test_loader else 0)
        
        # model in eval mode
        self.model.eval()

        # ------ Main test loop ------

        # do some stuff at the beginning of the training
        self._on_test_begin()

        if test_loader:        
            # we do not compute gradients in inference mode     
            with torch.no_grad(): 

                # loop over test data
                i = 0
                for data in test_loader:  

                    # push to device
                    images, labels = data[0].to(self.device), data[1].to(self.device)  

                    # forward pass through the network
                    outputs = self.model(images)
                   
                    # compute accuracy                
                    self.metrics["accuracy"] += self._compute_accuracy(outputs, labels)

                    # status update
                    if ((i % 100) == 0):
                        self.logger_stream_handler.terminator = ""
                        self.logger.info ('.')

                    i = i+1

         # do some stuff at the end of the testing
        self._on_test_end(num_test_samples)