from dataclasses import dataclass, field

@dataclass
class Staff:
    sort_index:int = field(init=False,repr=False)
    name: str
    age: int
    salary: float = "Not Defined"

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
    
    def salary_hike(self,percent=10) -> int:

        self.percent_scaled = percent/100

        hiked_salary = self.salary + self.salary*self.percent_scaled

        return f"{hiked_salary} 🤑"
        

if __name__ == "__main__":
       
    employee_obj = Employee("John",25,8000)
    print(employee_obj)
    
    print(employee_obj.salary_hike())
    print(employee_obj.percent_scaled)


    staff_obj = Staff("John",25,8000)

    print(staff_obj)


