import time
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

from tqdm import tqdm

from fileLoader import zaladujZdjecia
from generator import Generator

il_generacji = 3
il_osobnikow = 8
force_cpu = False

def main():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    if force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    parametry = {
        'neurony': [32, 64, 128],
        'aktywacja': ['elu', 'relu', 'sigmoid', 'tanh'],
        'il_warstw': [2, 3]
    }

    dane, etykiety = zaladujZdjecia()
    x, y = shuffle(dane, etykiety, random_state=2)
    il_etykiet = etykiety.shape[1]

    # dzielimy na dane testowe i trenujace
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)

    generator = Generator(parametry)
    print("[INFO]\tTworzenie wstepnej populacji")
    populacja = generator.stworzPopulacje(il_osobnikow, il_etykiet)

    for i in range(il_generacji):
        print("[INFO]\tGeneracja {}/{}".format(i + 1, il_generacji))
        print("[INFO]\tTrenowanie...")
        trenujSiec(populacja, x_train, x_test, y_train, y_test, il_etykiet)
        print("[INFO]\tWytrenowano, srednia celnosc: {:.4f}%".format(sredniaCelnosc(populacja) * 100))

        if(i < il_generacji - 1):
            print("[INFO]\tEwolucja")
            populacja = generator.ewoluuj(populacja)

    populacja = sorted(populacja, key=lambda x: x.celnosc, reverse=True)
    print("[INFO]\tCelnosc najlepszej sieci: {:.4f}%".format(populacja[0].celnosc * 100))

def sredniaCelnosc(sieci):
    celnosc = 0

    for siec in sieci:
        celnosc += siec.celnosc

    return celnosc / len(sieci)

def trenujSiec(sieci, x_train, x_test, y_train, y_test, il_etykiet):
    pasek = tqdm(total=len(sieci))

    for siec in sieci:
        siec.trenuj(x_train, x_test, y_train, y_test, il_etykiet)
        pasek.update(1)

    pasek.close()

if __name__ == '__main__':
    main()