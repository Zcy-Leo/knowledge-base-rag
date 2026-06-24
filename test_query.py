import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')
db = Chroma(persist_directory='./my_local_database', embedding_function=embeddings)

query = 'How to reset the device'
results = db.similarity_search(query, k=3)

print(f'Query: {query}')
print(f'Results: {len(results)} entries')
print('=' * 60)
for i, res in enumerate(results):
    meta = res.metadata
    print(f'[Result {i+1}]')
    print(f'  Type: {meta.get("type", "?")}')
    print(f'  Title: {meta.get("title", "?")}')
    print(f'  Page: {meta.get("source_page", "?")}')
    print(f'  Content:')
    content = res.page_content[:400]
    print(f'    {content}')
    print('=' * 60)
