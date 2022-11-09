class Progress:
    def __init__(self):
        self.status = "starting"

    def setStatus(self, newStatus):
        self.status = newStatus

    def getStatus(self):
        return self.status
