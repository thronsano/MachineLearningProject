from siec import Siec
import random

class Generator():
    def __init__(self, parametry):
        self.parametry = parametry
        self.pozostaw = 0.5
        self.szansa_pozostania = 0.05
        self.szansa_mutacji = 0.3

    def stworzPopulacje(self, il_osobnikow, il_etykiet):
        spis_osobnikow = []

        for i in range(il_osobnikow):
            osobnik = Siec(self.parametry)
            osobnik.wygeneruj()

            spis_osobnikow.append(osobnik)

        return spis_osobnikow

    def ewoluuj(self, populacja):
        ocenieni = [(siec.celnosc, siec) for siec in populacja]
        ocenieni = [x[1] for x in sorted(ocenieni, key=lambda x: x[0], reverse=True)]

        il_pozostalych = int(len(ocenieni) * self.pozostaw)
        rodzice = ocenieni[:il_pozostalych] # rodzice to ci ktorych zostawiamy do kolejnej generacji

        # pozostaw kilku losowych z odrzuconych
        for siec in ocenieni[il_pozostalych:]:
            if self.szansa_pozostania > random.random():
                rodzice.append(siec)

        il_rodzicow = len(rodzice)
        il_dopelnienia = len(populacja) - il_rodzicow
        dzieci = []

        while len(dzieci) < il_dopelnienia:
            tata = random.randint(0, il_rodzicow - 1)
            mama = random.randint(0, il_rodzicow - 1)

            if (tata != mama): # sprawdzamy czy tata i mama to nie jest ta sama siec
                tata = rodzice[tata]
                mama = rodzice[mama]

                nowe_dzieci = self.krzyzuj(mama, tata)

                for siec in nowe_dzieci:
                    if len(dzieci) < il_dopelnienia:
                        dzieci.append(siec)

        rodzice.extend(dzieci)
        return rodzice

    def krzyzuj(self, mama, tata):
        dzieci = []
        for i in range(2): #dwojka potomkow z dwojki rodzicow
            dziecko = {}

            for parametr in self.parametry:
                dziecko[parametr] = random.choice([mama.wybrane_parametry[parametr], tata.wybrane_parametry[parametr]])

            siec = Siec(self.parametry)
            siec.ustaw_parametry(dziecko)

            if self.szansa_mutacji > random.random():
                siec = self.mutuj(siec)

            dzieci.append(siec)
        return dzieci

    def mutuj(self, siec):
        mutowana_cecha = random.choice(list(self.parametry.keys()))
        siec.wybrane_parametry[mutowana_cecha] = random.choice(self.parametry[mutowana_cecha])
        return siec