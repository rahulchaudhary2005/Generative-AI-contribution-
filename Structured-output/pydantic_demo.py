from os import name

from pydantic import BaseModel, Field

class student(BaseModel):
    name: str
    
new_stuent = {'name': 'Rahul chaudhary'}
student=student(**new_stuent) 
print(student)  