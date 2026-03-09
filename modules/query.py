from modules.logger import logger


def query_chain(chain,user_input):
    try:
        logger.debug(f"Received user input: {user_input}")
        result = chain({"question": user_input})
        logger.debug(f"Query result: {result}")
        response={"answer": result['answer'], "source":[doc.metadata.get('source') for doc in result['source_documents']]}
        return response
    except Exception as e:
        logger.error(f"Error occurred while querying chain: {e}")
        raise