from typing import TypedDict

from torch import ne

class person(TypedDict):
    name: str
    age: int

new_person: person={
    'name': 'Rahul chaudhary',
    'age':20
}    
print(new_person)