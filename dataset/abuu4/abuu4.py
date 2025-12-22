import scipy.io as sio
import os


class abuu4:
    name = 'abu-urban-4'
    
    @staticmethod
    def get_data():
        path = os.path.dirname(__file__)
        file_name = os.path.join(path, 'abu-urban-4.mat')
        mat = sio.loadmat(file_name)
        data = mat['data'].astype(float)
        gt = mat['map'].astype(bool)
        return data, gt
