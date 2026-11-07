"""
# First, try to split along class definitions
"\nclass " --> Splitting based on class definitions in Python code. This will help in isolating each class and its methods for better readability and analysis.,
"\ndef " --> Splitting based on function definitions in Python code.,
"\n\tdef ", --> Splitting based on method definitions within classes.


#### Now split by the normal type of lines
"\n\n" --> Splitting based on double newlines to separate paragraphs or blocks of code.,
"\n" --> Splitting based on single newlines.,
" " --> Splitting based on spaces to isolate individual words or tokens.,
"" --> Splitting based on empty strings.,
"""


from langchain_text_splitters import RecursiveCharacterTextSplitter,Language

text = """
class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade  # Grade is a float (like 8.5 or 9.2)

    def get_details(self):
        return self.name"

    def is_passing(self):
        return self.grade >= 6.0


# Example usage
student1 = Student("Aarav", 20, 8.2)
print(student1.get_details())

if student1.is_passing():
    print("The student is passing.")
else:
    print("The student is not passing.")

"""
# Initialize the splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
)

# Perform the split
chunks = splitter.split_text(text)

print(len(chunks))
print(chunks[1])