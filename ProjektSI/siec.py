import random
import numpy as np
from keras_applications.vgg16 import VGG16
from keras.layers import Dense, Activation, Flatten
from keras.layers import merge, Input
from keras.models import Model
from sklearn.metrics import confusion_matrix

class Siec():
    def __init__(self, parametry):
        self.parametry = parametry
        self.wybrane_parametry = {}
        self.celnosc = 0
        self.strata = 0
        self.histogram = {}
        self.batch_size = 16
        self.epochs = 5
        self.verbose = 0
        self.confusion_matrixx = []

    def wygeneruj(self):
        for klucz in self.parametry:
            self.wybrane_parametry[klucz] = random.choice(self.parametry[klucz])

    def trenuj(self, x_train, x_test, y_train, y_test, il_etykiet):
        self.il_etykiet = il_etykiet

        #pobieramy parametry obecnej sieci neuronowej
        il_neuronow = self.wybrane_parametry['neurony']
        sposob_aktywacji = self.wybrane_parametry['aktywacja']
        il_warstw = self.wybrane_parametry['il_warstw']

        wejscie = Input(shape=(224, 224, 3))

        #tworzymy model bazujac na VGG16
        model = VGG16(input_tensor=wejscie, include_top=False, weights='imagenet')
        x = model.get_layer('block5_pool').output
        x = Flatten(name='flatten')(x)

        # dodajemy warstwy
        for i in range(il_warstw):
            x = Dense(il_neuronow, activation=sposob_aktywacji)(x)

        # warstwa wyjsciowa
        wyjscie = Dense(self.il_etykiet, activation='softmax')(x)
        nowy_model_vgg = Model(wejscie, wyjscie)

        # zamrazamy warstwy poza nasza nowa, zeby nie szkolic od nowa
        # orginalnego modelu
        for warstwa in nowy_model_vgg.layers[:-(il_warstw + 1)]:
            warstwa.trainable = False

        nowy_model_vgg.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        histogram = nowy_model_vgg.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose, validation_data=(x_test, y_test))

        # ewaluacja
        (loss, accuracy) = nowy_model_vgg.evaluate(x_test, y_test, batch_size=self.batch_size, verbose=self.verbose)
        y_pred = nowy_model_vgg.predict(x_test, batch_size=self.batch_size)

        self.confusion_matrixx = confusion_matrix(y_test.argmax(axis=-1), y_pred.argmax(axis=-1))
        self.histogram = histogram
        self.strata = loss
        self.celnosc = accuracy

    def ustaw_parametry(self, parametry):
        self.wybrane_parametry = parametry

    def opisz(self):
        print("Celnosc:\t{:.4f}%".format(self.celnosc * 100), "\nStrata:\t\t{:.4f}".format(self.strata), "\nNeurony:\t", self.wybrane_parametry['neurony'], "\nAktywacja:\t", self.wybrane_parametry['aktywacja'], "\nIlosc warstw:\t", self.wybrane_parametry['il_warstw'], "\nConfusion matrix:\n", self.confusion_matrixx)
