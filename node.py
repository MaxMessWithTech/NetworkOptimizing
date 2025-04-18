from connection import Connection

class Node:
    def __init__(self, index:int):
        self.index = index
        self.connections = list()
        
    def addConn(self, conn: Connection) -> bool:
        self.connections.append(conn)
        return True

    def makeRow(self, baseList:list[float]) -> list[float]:
        for conn in self.connections:
            baseList.insert(conn.index, conn.weight)

    def getConnections(self) -> list[Connection]:
        return self.connections
