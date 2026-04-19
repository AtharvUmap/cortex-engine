from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents: list) -> list:
    """Split documents into smaller chunks for embedding.

    Args:
        documents: A list of LangChain Document objects.

    Returns:
        A list of chunked Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return splitter.split_documents(documents)
