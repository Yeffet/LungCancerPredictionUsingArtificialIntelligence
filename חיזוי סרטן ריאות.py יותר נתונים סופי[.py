import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
import sklearn.neural_network

def euclidean_distance(p1, p2):
    d = 0.0
    for i in range(len(p1)):
        a = float(p1[i])
        b = float(p2[i])
        d += np.power((a-b),2)
    d = np.sqrt(d)
    return d

def takeSecond(elem):
    return elem[1]

def predict(train, test, lbl, K):
    distances = []
    for t, l in zip(train,lbl):
        dist = euclidean_distance(test, t)
        distances.append([t, dist, l])
    distances.sort(key=takeSecond)
    neighbors = []
    for i in range(K):
        neighbors.append(distances[i])
    out = [row[-1] for row in neighbors]
    return max(out,key=out.count)


heart_data = np.genfromtxt('survey lung cancer new no title.csv', delimiter=',') #uploads the file
np.random.shuffle(heart_data) #Shuffle the data



#Divides the data into groups
train_data = np.array(heart_data[:269,:15])
train_lbl = np.array(heart_data[:269,15])
test_data = np.array(heart_data[269:,:15])
test_lbl = np.array(heart_data[269:,15])
#print(train_data)
print("------------------")

    #Categorize the checklist
#print(heart_data)
lbl_pred = []
i = 0
corrects = 0
mistakes = 0
for i in range(len(test_data)):
    pred = predict(train_data, test_data[i], train_lbl, 3)
    if test_lbl[i] == pred:
        corrects += 1
    else:
        mistakes += 1
    lbl_pred.append(pred)



print(confusion_matrix(test_lbl, lbl_pred)) #Prints the table
print("corrects:", corrects, "mistakes:", mistakes)



#Receiving data from the user
print("--------------------------------------")
print("                                      ")
print("Enter certain data (that you have or from the Excel file attached to the code)")
print("which corresponds to lung cancer data and the project will output whether there is lung cancer or not.")
print("                                      ")
print("--------------------------------------")

print("YES=1, NO=0")
UserData = np.zeros(15)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>")
UserData[0] = input("input the GENDER(1=MALE, 0=FEMALE): ")
UserData[1] = input("input the AGE(44-87): ")
UserData[2] = input("SMOKING?: ")
UserData[3] = input("YELLOW_FINGERS?: ")
UserData[4] = input("ANXIETY?: ")
UserData[5] = input("PEER_PRESSURE?: ")
UserData[6] = input("CHRONIC DISEASE?: ")
UserData[7] = input("FATIGUE?: ")
UserData[8] = input("ALLERGY?: ")
UserData[9] = input("WHEEZING?: ")
UserData[10] = input("ALCOHOL?: ")
UserData[11] = input("COUGHING?: ")
UserData[12] = input("SHORTNESS OF BREATH?: ")
UserData[13] = input("SWALLOWING DIFFICULTY?: ")
UserData[14] = input("CHEST PAIN?: ")
print("--------------------------------------")

print("(1 = SICK, 0 = NOT_SICK) result:", int(predict(train_data, UserData, train_lbl, 3)))
