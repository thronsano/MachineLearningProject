import time
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.backend.tensorflow_backend import set_session
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from tqdm import tqdm
from fileLoader import zaladujZdjecia
from generator import Generator

il_generacji = 10
il_osobnikow = 16
force_cpu = False

def main():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    if force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    parametry = {
        'neurony': [32, 64, 128, 256, 512, 1024, 2048],
        'aktywacja': ['elu', 'relu', 'sigmoid', 'tanh'],
        'il_warstw': [2, 3, 4, 5, 6]
    }

    dane, etykiety = zaladujZdjecia()
    x, y = shuffle(dane, etykiety, random_state=2)
    il_etykiet = etykiety.shape[1]

    sredniaCelnoscTest = []
    sredniaStrataTest = []
    sredniaCelnoscTren = []
    sredniaStrataTren = []
    najlepszaCelnosc = []
    najlepszaStrata = []

    # dzielimy na dane testowe i trenujace
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)

    generator = Generator(parametry)
    print("[INFO]\tTworzenie wstepnej populacji")
    populacja = generator.stworzPopulacje(il_osobnikow, il_etykiet)

    for i in range(il_generacji):
        print("[INFO]\tGeneracja {}/{}".format(i + 1, il_generacji))
        print("[INFO]\tTrenowanie...")
        trenujSieci(populacja, x_train, x_test, y_train, y_test, il_etykiet)

        sredniaStrataTest.append(sredniaStrataTestujacy(populacja))
        sredniaCelnoscTest.append(sredniaCelnoscTestujacy(populacja)*100)
        sredniaStrataTren.append(sredniaStrataTrenujacy(populacja))
        sredniaCelnoscTren.append(sredniaCelnoscTrenujacy(populacja)*100)
        najlepszaCelnosc.append(najlepszaCelnoscSieci(populacja)*100)
        najlepszaStrata.append(najlepszaStrataSieci(populacja))

        print("[INFO]\tWytrenowano, srednia celnosc: {:.4f}%, srednia strata: {:.4f}".format(sredniaCelnoscTest[len(sredniaCelnoscTest) - 1], sredniaStrataTest[len(sredniaStrataTest) - 1]))

        if(i < il_generacji - 1):
            print("[INFO]\tEwolucja")
            populacja = generator.ewoluuj(populacja)

    populacja = sorted(populacja, key=lambda x: x.celnosc, reverse=True)
    
    print("[INFO]\t Najlepsza siec:")
    populacja[0].opisz()

    stworzGraf(sredniaStrataTest, sredniaCelnoscTest, sredniaStrataTren, sredniaCelnoscTren, najlepszaCelnosc, najlepszaStrata)

def sredniaCelnoscTestujacy(sieci):
    celnosc = 0

    for siec in sieci:
        celnosc += siec.celnosc

    return celnosc / len(sieci)

def sredniaStrataTestujacy(sieci):
    strata = 0

    for siec in sieci:
        strata += siec.strata

    return strata / len(sieci)

def sredniaCelnoscTrenujacy(sieci):
    celnosc = 0

    for siec in sieci:
        celnosc += sum(siec.histogram.history['acc']) / siec.epochs

    return celnosc / len(sieci)

def sredniaStrataTrenujacy(sieci):
    strata = 0

    for siec in sieci:
        strata += sum(siec.histogram.history['loss']) / siec.epochs

    return strata / len(sieci)

def najlepszaCelnoscSieci(sieci):
    najlepszaCelnosc = 0

    for siec in sieci:
        if siec.celnosc > najlepszaCelnosc:
            najlepszaCelnosc = siec.celnosc

    return najlepszaCelnosc

def najlepszaStrataSieci(sieci):
    najlepszaStrata = 999

    for siec in sieci:
        if siec.strata < najlepszaStrata:
            najlepszaStrata = siec.strata

    return najlepszaStrata

def trenujSieci(sieci, x_train, x_test, y_train, y_test, il_etykiet):
    pasek = tqdm(total=len(sieci))

    for siec in sieci:
        siec.trenuj(x_train, x_test, y_train, y_test, il_etykiet)
        pasek.update(1)

    pasek.close()

def stworzGraf(strataTest, celnoscTest, strataTren, celnoscTren, najlepszaCelnosc, najlepszaStrata):
    zakres = range(1, il_generacji + 1)

    plt.figure(1, figsize=(7,5))
    plt.plot(zakres, celnoscTest)
    plt.plot(zakres, celnoscTren)
    plt.plot(zakres, najlepszaCelnosc)
    plt.xlabel('Liczba generacji')
    plt.ylabel('Celnosc w %')
    plt.title('Graf dopasowania - srednia celnosc')
    plt.grid(True)
    plt.legend(['celnosc testujacy','celnosc trenujacy', 'celnosc najlepsza'])
    plt.style.use(['classic'])

    plt.figure(2, figsize=(7,5))
    plt.plot(zakres, strataTest)
    plt.plot(zakres, strataTren)
    plt.plot(zakres, najlepszaStrata)
    plt.xlabel('Liczba generacji')
    plt.ylabel('Strata')
    plt.title('Graf dopasowania - srednia strata')
    plt.grid(True)
    plt.legend(['strata testujacy','strata trenujacy', 'najmniejsza strata'])
    plt.style.use(['classic'])
    plt.show()

if __name__ == '__main__':
    main()