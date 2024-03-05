from datetime import datetime

class Time:
    def datetime(self):
        return str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    def date(self):
        return str(datetime.now().strftime("%Y-%m-%d"))