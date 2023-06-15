import numpy as np
import matplotlib.pyplot as plt

# Skapar alla mätta x-värden (alltså bara x-värden 0-99)
x = np.linspace(0, 99, 100)

# Skapar alla mätta y-värden (Dessa är y-värden från 0-99, men sen så sprider jag de lite med hjälp av en random array)
y = np.linspace(0, 99, 100)
z = np.random.rand(100)*30
y = y + z

# Det här är en array som innehåller information om spread och linjen som ska ritas på formen: (spridning, (lutning, intercept))
RMSE = (15000, (0, 0))

def computeValues(p1, p2):
    
    # Beräknar linjens ekvation för två inmatnings punkter p1 och p2
    dydx = (p2[1] - p1[1])/(p2[0] - p1[0])
    b = p2[1] - dydx*p2[0] 

    return (dydx, b)

"""Beräknar mean square error (spread i denna fil) givet linjen (fit)"""
def computeRMSE(fit):
    square_sum = 0
    # Liten kommentar: jag har skrivit 100 här för jag vet att det finns 100 datapunkter, men egentligen vill man ju ha n datapunkter
    for n in range(100):
        square_sum += ((fit[0]*x[n]+fit[1]) - y[n])**2
    return np.sqrt((square_sum/100))

"""Här körs koden för att välja p1 och p2 och sedan beräkna linjen samt dess spridning"""
for i in range(0, 99):
    # p1
    for j in range(i+1, 99):
        # p2
        p1 = (x[i], y[i])
        p2 = (x[j], y[j])
        fit = computeValues(p1, p2)
        spread = computeRMSE(fit)
        # Här sparar vi spridnignen och linjen, om vi mäter en lägre spridning omdefinierar vi RMSE till dessa nya bättre värden
        if(spread < RMSE[0]):
            RMSE = (spread, fit)

"""Vi beräknar nu linjen. (Du kan se här att jag helt enkelt bara har gångrat själva x-värdena, vilka innehålls i en numpy-array,
och det kanske bara är en liten reminder för dig men man kan ju bara gångra en hel array med ett värde för att få en ny array
precis som matris gånger skalär ger en en ny matris."""
y_predicted =  RMSE[1][0]*x + RMSE[1][1]
plt.scatter(x, y)
plt.plot(x, y_predicted)
plt.show()
