#подсчет времени работы функций
# Когда сам декоратор принимает аргументы
#Декоратор2 - с входными аргументами
from datetime import datetime
def timeit(arg):
    #print(arg)
    def outer(func):
        def wrapper(*args, **kwargs):
            start = datetime.now()
            result = func(*args, **kwargs)
            print("All time of " + str(arg) + " : " + str(datetime.now() - start))
            return result
        return wrapper
    return outer

def timeitNoPrint(arg):
    #print(arg)
    def outer(func):
        def wrapper(*args, **kwargs):
            start = datetime.now()
            result = func(*args, **kwargs)
            print("All time: " + str(datetime.now() - start))
            return result
        return wrapper
    return outer
