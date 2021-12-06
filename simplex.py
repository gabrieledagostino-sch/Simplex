import numpy as np
import math
import itertools

RELATIVETOLERANCE = 1e-9
ABSOLUTETOLERANCE = 1e-12


class constrainType():
        Equals = 0
        LessThan = 1
        GreaterThan = 2


class objFunction():
        def __init__(self, c, isMin):
                self.coef = np.array(c)
                self.coef *= (-1)**(isMin) #if it is a minimum multiply by -1 else do nothing
        
        def apply(self, x):
                return np.dot(self.coef, x)
        
        def __str__(self):
                s = "max "
                for i in (str(self.coef[i])+"x"+str(i)+" " for i in range(self.coef.size)):
                        s += i
                return s


class constraint():
        def __init__(self, A_i, b, type):
                self.coef = np.array(A_i)
                self.known = b
                self.type = type
        
        def isRespected(self, x):
                a = np.dot(self.coef, x)
                if(self.type == constrainType.LessThan):
                        return a <= self.known + RELATIVETOLERANCE*max(abs(a),abs(self.known))+ABSOLUTETOLERANCE
                if(self.type == constrainType.GreaterThan):
                        return a >= self.known - RELATIVETOLERANCE*max(abs(a),abs(self.known))-ABSOLUTETOLERANCE
                if(self.type == constrainType.Equals):
                        return math.isclose(a,self.known, rel_tol=RELATIVETOLERANCE, abs_tol=ABSOLUTETOLERANCE)
                
        def __str__(self):
                s = ""
                for i in (str(self.coef[i])+"x"+str(i)+" " for i in range(self.coef.size)):
                        s += i
                s += "<= " if self.type == constrainType.LessThan else ">= " if self.type == constrainType.GreaterThan else "= "
                s += str(self.known)
                return s

class PLProblem():
        def __init__(self, c, isMin, coefList, b, typeList):
                self.objF = objFunction(c, isMin)
                self.c = self.objF.coef

                for i in range(len(coefList)):
                        if typeList[i] == constrainType.Equals:
                                typeList.insert(i+1, constrainType.LessThan)
                                b.insert(i+1, b[i])
                                coefList.insert(i+1, coefList[i])
                                typeList[i] = constrainType.GreaterThan
                
                self.A = np.empty(shape=(len(b), len(c)))
                self.b = np.empty(shape=(len(b)))
                self.conList = list()

                for i in range(len(coefList)):
                        self.A[i] = np.array(coefList[i])
                        self.b[i] = b[i]
                        if typeList[i] == constrainType.GreaterThan:
                                self.A[i] *= -1
                                self.b[i] *= -1
                        
                        self.conList.append(constraint(self.A[i], self.b[i], constrainType.LessThan))
                
        def auxP(self):
                b = None

                for i in itertools.combinations(range(self.b.size), self.c.size):
                        subM = [self.A[x] for x in i]
                        if(np.linalg.det(subM) != 0):
                                b = list(i)
                                break
                if b == None:
                        return np.inf

                bb = [self.b[i] for i in range(self.b.size) if i in b]
                Ab = [self.A[i] for i in range(self.b.size) if i in b]
                
                Abi = np.linalg.inv(Ab)
                
                signedX = np.dot(Abi,bb)

                V = {i for i in range(self.b.size) if not self.conList[i].isRespected(signedX)}
                if len(V) == 0:
                        return (None,(0,[],b))
                
                variables =np.hstack((np.zeros(shape=(self.c.size)),np.ones(shape=(len(V)))))

                epsNum = 0
                conList = list()
                kw = list()
                for i in range(self.b.size):
                        vals = np.zeros(shape=(len(V)))
                        if epsNum < len(V) and i in V:
                                vals[epsNum] = -1
                                epsNum += 1
                        coef = np.hstack((self.A[i], vals))
                        conList.append(coef)
                        kw.append(self.b[i])

                b = b + list(V)

                for i in range(len(V)):
                        coef = np.zeros((self.c.size+ len(V)))
                        coef[i + self.c.size] = -1
                        conList.append(coef)
                        kw.append(0)
                Paux = PLProblem(variables, True, conList, kw, [constrainType.LessThan] * len(kw))
                return (Paux, b)
                

        def simplexPass(self,b):
                bb = [self.b[i] for i in range(self.b.size) if i in b]
                Ab = [self.A[i] for i in range(self.b.size) if i in b]
                Abi = np.linalg.inv(Ab)
                signedX = np.dot(Abi, bb)

                signedY = np.zeros(self.b.size)
                for i in range(len(b)):
                        signedY[b[i]] = np.dot(self.c, Abi[:,i])

                h = np.where(signedY < abs(signedY))[0]
                if h.size == 0:
                        return (self.objF.apply([i if i >= 0 + ABSOLUTETOLERANCE else 0 for i in signedX]), signedX, b)
                h = h[0]
                h = next(i for i in range(len(b)) if h == b[i])
                etha = np.inf
                k = 0

                W = -1*Abi
                Wh = W[:,h]
                for i in (x for x in range(self.b.size) if x not in b and np.dot(self.A[x], Wh) > 0+ABSOLUTETOLERANCE):
                        Ax = np.dot(self.A[i], signedX)
                        AWh = np.dot(self.A[i], Wh)
                        letha = (self.b[i] - Ax)/AWh
                        if letha <= etha + ABSOLUTETOLERANCE + RELATIVETOLERANCE*max(etha,letha):
                                etha = letha
                                k = i
                if etha == np.inf:
                        return np.inf
                
                b[h] = k
                return sorted(b)


        def simplex(self):
                Pauxb = self.auxP()
                if Pauxb == np.inf:
                        return None
                
                b = Pauxb[1]
                if type(b) == list:
                        b = sorted(b)
                Paux = Pauxb[0]
                while type(b) == list:
                        b = Paux.simplexPass(b)
                if b[0] > 0:
                        return None
                b = b[2]
                b = [i for i in b if i < self.b.size]
                b = sorted(b)
                while type(b) == list:
                        b = self.simplexPass(b)
                return b


        
        def __str__(self):
                s = ""
                s += str(self.objF)
                s += "\n"
                for i in self.conList:
                        s+=str(i)+"\n"
                return s
        
                






if __name__ == "__main__":
        PL1 = PLProblem([1,1,1], False, [[1,-1,1], [2,1,-2], [1,1,2], [4,1,1]], [1,1,4,6], [constrainType.LessThan]*4)
        print(PL1.simplex())
        PL2 = PLProblem([-5,-4,-8,-9,-15], True, [[1,1,2,1,2],[1,-2,-1,2,3],[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]], [11,6,0,0,0,0,0], [constrainType.Equals,constrainType.Equals,constrainType.GreaterThan,constrainType.GreaterThan,constrainType.GreaterThan,constrainType.GreaterThan,constrainType.GreaterThan])
        print(PL2.simplex())
        PL3 = PLProblem([1,1], False, [[2,-2],[2,-3],[1,2],[2,1]],[1,1,2,2], [constrainType.LessThan]*4)
        print(PL3.simplex())
        PL4 = PLProblem([6,5], False, [[1,2],[1,2],[1,0],[0,1]], [8,6,0,0], [constrainType.LessThan,constrainType.LessThan,constrainType.GreaterThan,constrainType.GreaterThan])
        print(PL4.simplex())
        PL5 = PLProblem([1,1], False, [[-1,1],[1,-6],[2,1]], [1,1,15],[constrainType.LessThan]*3)
        print(PL5.simplex())
        
        
