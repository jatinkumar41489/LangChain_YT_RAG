"""
1) First, try to split along Markdown headings (starting with level 2)

"\n#{1,6} " ---> Markdown headings (jaise #, ##, ### wagerah) ke base par split karne ke liye taaki alag-alag sections alag ho sakein.

# Note the alternative syntax for headings (below) is not handled here

2) Heading level 2
-----------
End of code block
"```\n" ---> Code blocks ke base par split karne ke liye taaki code aur text alag ho sakein.

3) Horizontal lines
"\n\\*\\*\\*+\n" ---> Asterisks wali horizontal divider lines (jaise ***) ke base par split karne ke liye.

"\n---+\n" ---> Hyphens wali horizontal divider lines (jaise ---) ke base par split karne ke liye.

"\n___+\n" ---> Underscores wali horizontal divider lines (jaise ___) ke base par split karne ke liye.

# Note that this splitter doesn't handle horizontal lines defined
by three or more of ***, ---, or ___, but this is not handled

4) Now split by the normal type of lines

"\n\n" ---> Double newlines (paragraphs ya bade line gaps) ke base par split karne ke liye.

"\n" ---> Single newlines (normal line breaks) ke base par split karne ke liye.

" " ---> Spaces ke base par split karne ke liye taaki individual words ya tokens alag ho sakein.

"" ---> Empty strings (single characters) ke base par split karne ke liye jab text max size se chhota karna ho.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
Space exploration has led to incredible scientific discoveries. From landing on the Moon to exploring Mars, humanity continues to push the boundaries of what’s possible beyond our planet.

These missions have not only expanded our knowledge of the universe but have also contributed to advancements in technology here on Earth. Satellite communications, GPS, and even certain medical imaging techniques trace their roots back to innovations driven by space programs.
"""

# Initialize the splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len #Chunk ki length napne ke liye standard Python character length (len()) ka use hoga. Hum len() ki jagah token ka use bhi kar sakte hai isse humare chunks token ke hisaab se banenge.
)

# perform the split
chunks = splitter.split_text(text)

print(len(chunks))
print(chunks[0])

