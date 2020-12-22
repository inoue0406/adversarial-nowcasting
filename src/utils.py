import csv
import numpy as np
import pandas as pd

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()

def log_freq_distr_pytorch(x,fname):
    # this routine logs the frequency of particular tensor
    x_np = x.to('cpu').detach().numpy().copy()
    print("tensor output shape:",x_np.shape)

    # generate histogram
    hist = np.histogram(x_np,bins=50)
    hist_df= pd.DataFrame({"value":hist[1][1:],
                           "count":hist[0]})
    hist_df.to_csv(fname)
    import pdb;pdb.set_trace()


    
    
