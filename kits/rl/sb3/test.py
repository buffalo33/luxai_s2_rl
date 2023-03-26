class Voiture:
    couleur = "rouge"

    def __init__(self):
        self.roue = 4
        self.capot = 1
        self.volant = 3

    def hello():
        print("hello wolrd")

voit = Voiture()
print(voit.__dict__.keys())


