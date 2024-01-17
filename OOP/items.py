from dataclasses import dataclass, field
import re

@dataclass
class Staff:
    sort_index:int = field(init=False,repr=False)
    name: str
    age: int
    salary: float = 0.00

    def __post_init__(self):
        self.sort_index = self.age
    def __repr__(self) -> str:
        
        return f"Staff - 🧑 {self.name.upper()}, {self.age}, ${self.salary} 💸"

class Employee:
    def __init__(self,name:str,age:int,salary:float) -> None:
        
        assert age >= 0, "Age should be positive integer"
        
        self.name = name
        self.age = age
        self.salary = salary
    
    def __repr__(self) -> str:
        
        return f"Employee - 🧑 {self.name.upper()}, {self.age}, ${self.salary} 💸"
    
    def salary_hike(self,percent=10) -> str:

        self.percent_scaled = percent/100

        self.salary = self.salary + self.salary*self.percent_scaled

        return f"{self.name} new salary: {self.salary} 🤑"
    
    def __lt__(self,other) -> bool:
        return (self.salary < other.salary)
    
    def __eq__(self,other) -> bool:

        return (self.name == other.name and self.age == other.age)
        

if __name__ == "__main__":
       
    employee_obj_1 = Employee("John",25,8000)
    employee_obj_2 = Employee("Kay",25,1000)

    print(employee_obj_1)
    print(employee_obj_1.salary_hike())

    if employee_obj_1 == employee_obj_2:
        print(f"Duplicate Entry")

    sorted_staff_list = sorted([employee_obj_1,employee_obj_2])
    print(sorted_staff_list)

    staff_obj = Staff("John",25,8000)
    print(staff_obj)