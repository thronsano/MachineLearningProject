import numpy as np
import os

from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from keras.utils import np_utils

def zaladujZdjecia():
    # ladujemy dane z katalogu 'data'
    SCIEZKA = os.getcwd()
    sciezka_do_data = SCIEZKA + '/data'
    lista_folderow = os.listdir(sciezka_do_data)

    lista_zdjec = []

    # lista poczatkowych numerow zdjec w folderach
    poczatkowe_zdjecia = []
    # lista ostatnich numerow zdjec w folderach
    koncowe_zdjecia = []

    # wczytujemy dane
    for iterator, zbior in enumerate(lista_folderow):
        # wczytujemy katalogi z folderu 'data'
        lista_zdjec_w_folderze = os.listdir(sciezka_do_data + '/' + zbior)
        print('[INFO]\tLoaded the images of dataset-' + '{}'.format(zbior))

        # przypisujemy dane dot.  poczatkow i koncow plikow w folderach
        if iterator != 0:
            poczatkowe_zdjecia.append(koncowe_zdjecia[iterator - 1] + 1)
            koncowe_zdjecia.append(poczatkowe_zdjecia[iterator] + len(lista_zdjec_w_folderze) - 1)
        else:
            poczatkowe_zdjecia.append(0)
            koncowe_zdjecia.append(len(lista_zdjec_w_folderze) - 1)

        # wczytujemy zdjecia
        for zdj in lista_zdjec_w_folderze:
            zdj_sciezka = sciezka_do_data + '/' + zbior + '/' + zdj
            zdj = image.load_img(zdj_sciezka, target_size=(224, 224))
            x = image.img_to_array(zdj)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            lista_zdjec.append(x)


    # przeksztalcamy dane, aby pozbyc sie zbednych informacji
    zdj_dane = np.array(lista_zdjec)
    zdj_dane = np.rollaxis(zdj_dane,1,0)
    zdj_dane = zdj_dane[0]

    # zbieramy etykiety na podstawie nazw folderow
    spis_etykiet = []
    il_roznych_etykiet = len(lista_folderow)
    ilosc_probek = zdj_dane.shape[0]
    etykiety_zdjec = np.ones((ilosc_probek,),dtype='int64')


    # przypisujemy etykiety do obrazkow pomagajac sobie iloscia plikow w folderze
    for i, zbior in enumerate(lista_folderow):
        etykiety_zdjec[poczatkowe_zdjecia[i]:koncowe_zdjecia[i]] = i
        spis_etykiet.append(zbior)

    # zamien dane na wektor one-hot
    Y = np_utils.to_categorical(etykiety_zdjec, il_roznych_etykiet)

    return zdj_dane, Y
