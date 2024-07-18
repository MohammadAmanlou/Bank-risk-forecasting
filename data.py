


class Data:
    def __init__(self , data):
        self.data = data
        self.uniqueVals = {}
    
    def addUniqueVals(self , key , value):
        self.uniqueVals[key] = value
        
    def getColumns(self):
        return self.data.columns
    
    
        
        
        