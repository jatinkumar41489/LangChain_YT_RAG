from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("dl-curriculum.pdf")

# Load the PDF document into memory.
docs = loader.load()

# Initialize the splitter
splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0,
    separator=""
)

# Split documents
result = splitter.split_documents(docs)

print(result[1].page_content)