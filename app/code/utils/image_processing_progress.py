class ImageProcessingProgress:
    def __init__(self):
        self.status = "starting"
        self.cause = ""

    def setStatus(self, newStatus):
        self.status = newStatus

    def setCause(self, newCause):
        self.cause = newCause

    def getStatus(self):
        return self.status

    def getCause(self):
        return self.cause
