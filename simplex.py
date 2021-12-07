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

                bb = [self.b[i] for i in b]
                Ab = [self.A[i] for i in b]
                
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
                

        def primalSimplexPass(self,b):
                bb = [self.b[i] for i in b]
                Ab = [self.A[i] for i in b]
                Abi = np.linalg.inv(Ab)
                signedX = np.dot(Abi, bb)

                signedY = np.zeros(self.b.size)
                for i in range(len(b)):
                        signedY[b[i]] = np.dot(self.c, Abi[:,i])

                h = np.where(signedY < abs(signedY) - ABSOLUTETOLERANCE)[0]
                if h.size == 0:
                        return (self.objF.apply([i if i >= 0 + ABSOLUTETOLERANCE else 0 for i in signedX]), signedX, b)
                h = h[0]
                h = next(i for i in range(len(b)) if h == b[i])
                tetha = np.inf
                k = 0

                W = -1*Abi
                Wh = W[:,h]
                for i in (x for x in range(self.b.size) if x not in b and np.dot(self.A[x], Wh) > 0+ABSOLUTETOLERANCE):
                        Ax = np.dot(self.A[i], signedX)
                        AWh = np.dot(self.A[i], Wh)
                        ltetha = (self.b[i] - Ax)/AWh
                        if ltetha < tetha:
                                tetha = ltetha
                                k = i
                if tetha == np.inf:
                        return np.inf
                
                b[h] = k
                return sorted(b)


        def primalSimplex(self):
                Pauxb = self.auxP()
                if Pauxb == np.inf:
                        return None
                
                b = Pauxb[1]
                if type(b) == list:
                        b = sorted(b)
                Paux = Pauxb[0]
                while type(b) == list:
                        b = Paux.primalSimplexPass(b)
                if b[0] > 0:
                        return None
                b = b[2]
                b = [i for i in b if i < self.b.size]
                b = sorted(b)
                while type(b) == list:
                        b = self.primalSimplexPass(b)
                return b

        
        def calcDual(self):
                self.dual = self.Dual(self.b, self.c, self.A)
        
        class Dual():
                def __init__(self,b,c,A):
                        self.b = b
                        self.c = c
                        self.A = A
                        self.objF = objFunction(b, True)
                        conList = list()
                        for i in range(c.size):
                                conList.append(constraint(A[:,i], c[i], constrainType.Equals))
                        for i in range(self.b.size):
                                coef = [0]*self.b.size
                                coef[i] = 1
                                conList.append(constraint(coef, 0, constrainType.GreaterThan))
                        self.conList = conList
                
                def __str__(self):
                        s = ""
                        s += str(self.objF)
                        s += "\n"
                        for i in self.conList:
                                s+=str(i)+"\n"
                        return s


                def auxD(self):                     
                        variables = np.hstack((np.zeros(shape=(self.b.size)),np.ones(shape=(self.c.size))))
                        for i in range(self.c.size):
                                if self.c[i] < 0:
                                        self.c[i] *= -1
                                        self.A[:,i] *= -1
                        aI = np.identity(self.c.size)
                        auxA = np.vstack((self.A,aI))
                        
                        Daux = PLProblem.Dual(variables, self.c, auxA)

                        base = [i + self.b.size for i in range(self.c.size)]
                        return (Daux, base)


                def dualSimplexPass(self,b):            #TOFIX something is broken with the tetha calculus
                        bb = [self.b[i] for i in b]
                        Ab = [self.A[i] for i in b]
                        Abi = np.linalg.inv(Ab)
                        signedX = np.dot(Abi,bb)
                        signedY = np.zeros(self.b.size)
                        for i in range(len(b)):
                                signedY[b[i]] = np.dot(self.c, Abi[:,i])
                        
                        biAx = self.b - np.dot(self.A,signedX)
                        k = np.where(biAx < abs(biAx) - ABSOLUTETOLERANCE)[0]
                        k = [x for x in k if x not in b]
                        
                        if len(k) == 0:
                                return (-1*self.objF.apply(signedY), signedY, b)
                        
                        k = k[0]

                        W = -1 * Abi
                        tetha = np.inf
                        h = None
                        for i in (x for x in range(len(b)) if np.dot(self.A[k], W[:,x]) < 0 - ABSOLUTETOLERANCE):
                                yi = signedY[b[i]]
                                akwi = -1*np.dot(self.A[k], W[:,i])
                                ltetha = yi / ( akwi)
                                if ltetha < tetha:
                                        h = i
                                        tetha = ltetha
                        
                        if h is None:
                                return np.inf
                        b[h] = k
                        return sorted(b)




                def dualSimplex(self):                 
                        aux = self.auxD()
                        baux = aux[1]
                        daux = aux[0]
                        if type(baux) == list:
                                baux = sorted(baux)
                        while type(baux) == list:
                                baux = daux.dualSimplexPass(baux)
                        
                        if baux == np.inf or baux[0] >= 0+ABSOLUTETOLERANCE:
                                return None
                        
                        baux= baux[2]
                        A2 = [daux.A[i] for i in range(daux.b.size) if i not in baux]
                        ME = [daux.A[i] for i in baux if i >= self.b.size]
                        ME = np.transpose(ME)
                        while len(ME) != 0:
                                h = None
                                k = None
                                for i in len(A2):
                                        for j in len(ME):
                                                if np.dot(A2[i], ME[j]):
                                                        k = i
                                                        h = j
                                                        break
                                        if h is not None:
                                                break
                                bh = baux.index(h)
                                baux[bh] = h
                        b = sorted(baux)
                        while type(b) == list:
                                b = self.dualSimplexPass(b)
                        return b
                        
        def dualSimplex(self):
                self.calcDual()
                sol = self.dual.dualSimplex()
                if sol == None :
                        return np.inf
                if sol == np.inf:
                        return None
                return sol
        

        
        def __str__(self):
                s = ""
                s += str(self.objF)
                s += "\n"
                for i in self.conList:
                        s+=str(i)+"\n"
                return s






if __name__ == "__main__":
        PL1 = PLProblem([1,1,1], False, [[1,-1,1], [2,1,-2], [1,1,2], [4,1,1]], [1,1,4,6], [constrainType.LessThan]*4)
        PL2 = PLProblem([-5,-4,-8,-9,-15], True, [[1,1,2,1,2],[1,-2,-1,2,3],[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]], [11,6,0,0,0,0,0], [constrainType.Equals,constrainType.Equals,constrainType.GreaterThan,constrainType.GreaterThan,constrainType.GreaterThan,constrainType.GreaterThan,constrainType.GreaterThan])
        PL3 = PLProblem([1,1], False, [[2,-2],[2,-3],[1,2],[2,1]],[1,1,2,2], [constrainType.LessThan]*4)
        PL4 = PLProblem([6,5], False, [[1,2],[1,2],[1,0],[0,1]], [8,6,0,0], [constrainType.LessThan,constrainType.LessThan,constrainType.GreaterThan,constrainType.GreaterThan])
        PL5 = PLProblem([1,1], False, [[-1,1],[1,-6],[2,1]], [1,1,15],[constrainType.LessThan]*3)
        PL6 = PLProblem([3,-4], False, [[-1,0],[0,-1],[1,2],[1,1],[1,0]], [0,0,13,9,7], [constrainType.LessThan]*5)
        print(PL1.primalSimplex())
        print(PL2.primalSimplex())
        print(PL3.primalSimplex())
        print(PL4.primalSimplex())
        print(PL5.primalSimplex())
        print(PL6.primalSimplex())
        print("dual")
        print(PL1.dualSimplex())
        print(PL2.dualSimplex())
        print(PL3.dualSimplex())
        print(PL4.dualSimplex())
        print(PL5.dualSimplex())
        print(PL6.dualSimplex())
        