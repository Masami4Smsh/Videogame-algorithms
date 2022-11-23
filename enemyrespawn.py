##To make this work, pls install the packages python -m pip install fuzzylogic

from matplotlib import pyplot
pyplot.rc("figure", figsize=(10, 10))

from fuzzylogic.classes import Domain, Set, Rule
from fuzzylogic.functions import R, S, triangular, trapezoid
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.neural_network import MLPClassifier

pyplot.subplot(211)
x = Domain("X Position", 0, 16)
x.left = S(5, 7)
x.right = R(9,11)
x.left.plot()
x.right.plot()

pyplot.subplot(212)
y = Domain("Y Position", 0, 12)
y.down = S(3, 5)
y.up = R(7,9)
y.down.plot()
y.up.plot()

pos = Domain("Spot", 0, 100)
pos.one = S(10,30)
pos.two = trapezoid(25, 30, 55, 60, c_m=0.9)
pos.three = trapezoid(40, 45, 70, 75, c_m=0.9)
pos. four = R(70,90)

R1 = Rule({(x.right, y.down): pos.one})
R2 = Rule({(x.left, y.down): pos.two})
R3 = Rule({(x.right, y.up): pos.three})
R4 = Rule({(x.left, y.up): pos.four})

rules = Rule({(x.right, y.down): pos.one,
              (x.left, y.down): pos.two,
              (x.right, y.up): pos.three,
              (x.left, y.up): pos.four,
             })

rules == R1 | R2 | R3 | R4 == sum([R1, R2, R3, R4])

values = {x: 10, y: 9}
print(R1(values), R2(values), R3(values), R4(values), "=>", rules(values))

stage = int(rules(values))

##Fuzzy logic no.2

cat = Domain("Difficulty", 0, 7)
spot = Domain("Midgame", 0, 100)
tim = Domain("Survival", 0, 240)

cat.easy = S(2, 4)
cat.middle = triangular(2, 6)
cat.hard = R(4, 6)

spot.early = S(30, 45)
spot.middle = triangular(25, 75)
spot.end = R(55, 70)

tim.short = S(30, 100)
tim.normal = triangular(80, 160)
tim.long = R(140, 200)

F1 = Rule({(cat.easy, spot.early): tim.long})
F2 = Rule({(cat.easy, spot.middle): tim.long})
F3 = Rule({(cat.easy, spot.end): tim.long})
F4 = Rule({(cat.middle, spot.early): tim.normal})
F5 = Rule({(cat.middle, spot.middle): tim.normal})
F6 = Rule({(cat.middle, spot.end): tim.normal})
F7 = Rule({(cat.hard, spot.early): tim.short})
F8 = Rule({(cat.hard, spot.middle): tim.short})
F9 = Rule({(cat.hard, spot.end): tim.short})

rulesf = Rule({(cat.easy, spot.early): tim.long,
              (cat.easy, spot.middle): tim.long,
              (cat.easy, spot.end): tim.long,
              (cat.middle, spot.early): tim.normal,
              (cat.middle, spot.middle): tim.normal,
              (cat.middle, spot.end): tim.normal,
              (cat.hard, spot.early): tim.short,
              (cat.hard, spot.middle): tim.short,
              (cat.hard, spot.end): tim.short,
             })

rulesf == F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 == sum([F1, F2, F3, F4, F5, F6, F7, F8, F9])

valuestwo = {cat: 3, spot : stage}


print(F1(valuestwo), F2(valuestwo), F3(valuestwo), F4(valuestwo),F5(valuestwo), 
F6(valuestwo), F7(valuestwo), F8(valuestwo), F9(valuestwo), "=>", rulesf(valuestwo))
##pyplot.show()
survival = int(rulesf(valuestwo))
## Neural Network

nom_archivo="tiempos_supervivencia.csv"
datos=np.loadtxt(nom_archivo, delimiter=",")
print(datos.shape) 

m=datos[:,0:6] #Caracteristicas del conjunto de datos
n=datos[:,6] #Etiquetas (clases)

print("=== ORIGINAL (primeras lineas) ===")
print(m[:5]) #Imprime las primera cinco lineas del data set
print(n[:5]) #Imprime las primeras cinco etiquetas

#Conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(m, n, test_size=.3, random_state=42)

print(X_train.shape) #Tamaño del conjunto de entrenamiento
print(X_test.shape) #Tamaño del conjunto de prueba

#Estandarizacion de los conjuntos de entrenamiento y prueba
X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test) 

print("=== ESTANDARIZADO ===")
print(" +++Entrenamiento+++")
print(X_train[:5])
print(" +++Prueba+++")
print(X_test[:5])

clf=MLPClassifier(random_state=42,max_iter=2000)
clf.fit(X_train, y_train)

accuracy=round(clf.score(X_test, y_test),3)
print("***Exactitud: ",accuracy,"***")

datos_nuevos=np.array([2,0,1,1,survival,100]).reshape(1,-1)

clase_predicha=clf.predict(datos_nuevos)
print(clf.predict_proba(datos_nuevos))

print("La clase del dato nuevo es:",clase_predicha)