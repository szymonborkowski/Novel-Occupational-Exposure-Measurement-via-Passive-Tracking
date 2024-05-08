class Doctor:
    def __init__(self, name, id):
        self.name = name
        self.occupational_exposure = 0
        self.id = id
        
    def add_occupational_exposure(self, amount):
        self.occupational_exposure += amount

    def __str__(self):
        return f"Staff details: Name - {self.name}, Occupational Exposure - {self.occupational_exposure} mSv/year"
