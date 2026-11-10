# ============================================================
# RAG using LangChain (YouTube Transcript)
# Flow: Indexing (1a-1d) -> Retrieval (2) -> Augmentation (3)
#       -> Generation (4) -> Chain
# ============================================================
# Install once in terminal:
# pip install youtube-transcript-api langchain-community langchain-huggingface \
#             langchain-text-splitters faiss-cpu sentence-transformers transformers torch

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    RequestBlocked,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline


# ============================================================
# Step 1a - Indexing (Document Ingestion)
# ============================================================

video_id = "Gfr50f6ZBvo"  # only the ID, not full URL

try:
    ytt_api = YouTubeTranscriptApi()
    transcript_list = ytt_api.fetch(video_id, languages=["en"]).to_raw_data()

    # Flatten it to plain text
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    print("Transcript fetched successfully!")
    print(transcript[:1000])

except RequestBlocked:
    raise RuntimeError("YouTube blocked requests from your IP. Try again later.")
except TranscriptsDisabled:
    raise RuntimeError("Captions are disabled for this video.")
except NoTranscriptFound:
    raise RuntimeError("English transcript was not found.")


# ============================================================
# Step 1b - Indexing (Text Splitting)
# ============================================================

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

print("\nTotal chunks:", len(chunks))
print("Sample chunk:", chunks[min(100, len(chunks) - 1)])


# ============================================================
# Step 1c & 1d - Indexing (Embedding Generation and Storing in Vector Store)
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

vector_store = FAISS.from_documents(chunks, embeddings)
print("FAISS Vector Store created successfully!")

print("Stored doc ids (first 3):", list(vector_store.index_to_docstore_id.values())[:3])

# Optional: save index so you don't have to rebuild it every run
vector_store.save_local("youtube_faiss_index")


# ============================================================
# Step 2 - Retrieval
# ============================================================

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

test_docs = retriever.invoke("What is this video about?")
print("\nRetrieved", len(test_docs), "chunks for test query")


# ============================================================
# Step 3 - Augmentation
# ============================================================

pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=256,
    do_sample=False,
)
llm = HuggingFacePipeline(pipeline=pipe)
print("Hugging Face LLM loaded!")

prompt = PromptTemplate(
    template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

{context}
Question: {question}
""",
    input_variables=["context", "question"],
)

question = "What is this video about?"
retrieved_docs = retriever.invoke(question)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

final_prompt = prompt.invoke({"context": context_text, "question": question})
print("\nFinal prompt (first 500 chars):")
print(final_prompt.to_string()[:500])


# ============================================================
# Step 4 - Generation
# ============================================================

answer = llm.invoke(final_prompt)
print("\nQuestion:", question)
print("Answer:", answer)


# ============================================================
# Building a Chain
# ============================================================

def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough(),
})

parallel_chain.invoke('who is Demis')

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser

print("\nChain test:")
print(main_chain.invoke("Can you summarize the video"))


# ============================================================
# Interactive Q&A (terminal)
# ============================================================

while True:
    user_q = input("\nAsk your question (type exit to stop): ")

    if user_q.strip().lower() == "exit":
        print("Goodbye!")
        break

    print("\nAnswer:")
    print(main_chain.invoke(user_q))