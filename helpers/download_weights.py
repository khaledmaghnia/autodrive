#Taken from: https://stackoverflow.com/a/53877507/13558274

import urllib.request
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download(tiny=False):

    print("Yolo weights not found in 'files/yolo*.weights'... downloading now...\n")

    output_path = 'files/yolov3.weights'
    url = 'https://pjreddie.com/media/files/yolov3.weights'
    if tiny:
        output_path = 'files/yolov3-tiny.weights'
        url = 'https://pjreddie.com/media/files/yolov3-tiny.weights'
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)