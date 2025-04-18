class Connection:
    def __init__(self, weight:int, node_1, node_2):
        self.weight = weight

        if node_1 == node_2:
            raise TypeError("NODE IS LOOPING TO ITSELF")

        self.node_1 = node_1
        self.node_2 = node_2

        self.node_1.addConn(self)
        self.node_2.addConn(self)
        
    def getOtherNode(self, node):
        if self.node_1 is node and self.node_2 is not node:
            return self.node_2
        
        if self.node_1 is not node and self.node_2 is node:
            return self.node_1
        
        raise TypeError("NODE IS LOOPING TO ITSELF")
    
    def __str__(self) -> str:
        return f"{self.weight}"
    
    def __repr__(self) -> str:
        return f"{self.weight}"
