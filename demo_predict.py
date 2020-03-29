import os

from seq_tag.predict import predict
from seq_tag.hparams import hparams

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
here = os.path.dirname(os.path.abspath(__file__))


def main():
    predict(hparams)


if __name__ == '__main__':
    main()
