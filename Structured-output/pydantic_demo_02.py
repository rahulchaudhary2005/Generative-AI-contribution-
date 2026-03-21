#default value in pydantic model

from pydantic import BaseModel,EmailStr
from typing import Optional

class student(BaseModel):
    name: str='Rahul chaudhary'
    age: Optional[int]=None
    email: Optional[EmailStr]=None

new_student={'age': 20, 'email':'rahul.chaudhary@example.com'}

student=student(**new_student)
student_json=student.model_dump_json()

print(student.email)    