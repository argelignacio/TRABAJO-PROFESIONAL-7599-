from datetime import datetime

class Time:
    def datetime(self):
        return str(datetime.now().strftime("%Y-%m-%d %H:%m:%S"))
    
    def date(self):
        return str(datetime.now().strftime("%Y-%m-%d"))