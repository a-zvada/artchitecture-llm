import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# === Загрузка документов ===
docs = []
knowledge_dir = "./knowledge_base"

for filename in os.listdir(knowledge_dir):
    if filename.endswith(".txt"):
        filepath = os.path.join(knowledge_dir, filename)
        loader = TextLoader(filepath, encoding="utf-8")
        loaded = loader.load()
        for doc in loaded:
            doc.metadata["source"] = filename
        docs.extend(loaded)


print(f"Загружено {len(docs)} документов")

# === Разбиение на чанки ===
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # ~120–150 слов
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)
splits = text_splitter.split_documents(docs)
print(f"Разбито на {len(splits)} чанков")

model_name = "BAAI/bge-m3"
#model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": True}

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
#    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

vectorstore_path = "./vectorstore/chroma_db"

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=vectorstore_path
)

print(f"Векторный индекс сохранён в: {vectorstore_path}")