from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

# use of lazy_load() --> returns a generator object, but load() return list of document
docs = loader.lazy_load()

print(docs[150].page_content)
print(docs[150].metadata)

for document in docs:
    print(document.metadata)