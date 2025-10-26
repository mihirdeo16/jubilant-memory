from functools import reduce
from math import pow

class MeanCalculator:
    def __init__(self,values:list) -> None:
        self.elements= values
    
    def __str__(self) -> str:
        
        return f"Elements of the array {self.elements}"
    
    def arithmetic_mean(self) -> float:
        mean = sum(self.elements)/len(self.elements)
        return mean

    def harmonic_mean(self) -> float:
        " This mean is AIM at units related to rate, e.g Speed - dist/time,F1 Score  "
        mean = len(self.elements)/sum([(1/x) for x in self.elements])
        return mean

    def geometric_mean(self) -> float:
        "Finding the average of percentages, ratios, indexes, or growth rates. "
        mean = pow(reduce((lambda x,y: x*y),self.elements),(1/len(self.elements)))

        return mean

    def quadratic_mean(self) -> float:

        mean = pow((sum([x**2 for x in self.elements]),len(self.elements)),2)
        
        return mean

    def weighted_mean(self) -> float:
        pass

    def exponential_weighted_mean(self) -> float:
        pass

if __name__ =="__main__":
    """


    """

    mean_type = int(input("Choose the type of mean you want to calculate? \n 1) Arithmetic mean (simple mean) \n 2) Geometric Mean \n 3) Harmonic mean \n 4) Quadratic mean \n 5) Weighted mean \n 6) Exponential weighted mean \n > "))

    values = list(map(float,input("Provide values (Space separated value) > ").split(" ")))

    mean_cal_obj = MeanCalculator(values)

    results = {
        1:mean_cal_obj.arithmetic_mean(),
        2:mean_cal_obj.geometric_mean(),
        3:mean_cal_obj.harmonic_mean(),
        4:mean_cal_obj.quadratic_mean(),
        5:mean_cal_obj.weighted_mean(),
        6:mean_cal_obj.exponential_weighted_mean(),
    }

    result = results[results[int(mean_type)]]

    print(result)

