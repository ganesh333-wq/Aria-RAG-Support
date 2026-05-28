"""
──────────────────────────────────────────────────────────────────────────────
 RAG Chain — LangChain ConversationalRetrievalChain wrapper

 This module demonstrates how to wire up LangChain's ConversationalRetrievalChain
 with ChromaDB and session memory. It is used as a reference implementation
 and for direct chain invocation when fine-grained pipeline control is needed.

 The main pipeline (pipeline_service.py) uses this chain for:
   - LangChain-native retrieval (as_retriever)
   - Automatic question condensation for follow-ups
   - ConversationBufferMemory integration

 For production requests, pipeline_service.py calls llm_service.generate_response()
 which uses LCEL chains directly for maximum control over:
   - Confidence filtering before LLM call
   - Escalation detection before LLM call
   - Groq-only generation with safe template fallback
──────────────────────────────────────────────────────────────────────────────
"""

from typing import Any, Dict, List, Optional

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever

from app.prompts.templates import ARIA_IDENTITY


# ── Prompt for the QA chain ───────────────────────────────────────────────────

QA_CHAIN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", ARIA_IDENTITY + "\n\nContext from knowledge base:\n{context}"),
    ("human", "{question}"),
])

# ── Condense question prompt (for follow-up handling) ─────────────────────────

CONDENSE_QUESTION_PROMPT = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=(
        "Given the chat history below and a follow-up question, "
        "rephrase the follow-up question to be a standalone question.\n\n"
        "Chat History:\n{chat_history}\n\n"
        "Follow-up Question: {question}\n"
        "Standalone Question:"
    ),
)


class AriaRAGChain:
    """
    LangChain ConversationalRetrievalChain for Aria.

    Provides a high-level interface that:
      - Manages per-session ConversationBufferMemory
      - Uses ChromaDB retriever for document lookup
      - Automatically rewrites follow-up questions via the condense prompt
      - Returns answer + source documents
    """

    def __init__(
        self,
        llm: BaseChatModel,
        retriever: VectorStoreRetriever,
    ):
        self.llm = llm
        self.retriever = retriever
        # Per-session chain instances (each needs its own memory)
        self._session_chains: Dict[str, ConversationalRetrievalChain] = {}

    def get_chain(self, session_id: str) -> ConversationalRetrievalChain:
        """
        Get or create a ConversationalRetrievalChain for the given session.
        Each session has isolated ConversationBufferMemory.
        """
        if session_id not in self._session_chains:
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer",
            )
            chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.retriever,
                memory=memory,
                condense_question_prompt=CONDENSE_QUESTION_PROMPT,
                combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
                return_source_documents=True,
                verbose=False,
            )
            self._session_chains[session_id] = chain

        return self._session_chains[session_id]

    async def ainvoke(self, session_id: str, question: str) -> Dict[str, Any]:
        """
        Invoke the chain asynchronously for a given session and question.

        Returns:
            { "answer": str, "source_documents": List[Document] }
        """
        chain = self.get_chain(session_id)
        result = await chain.ainvoke({"question": question})
        return {
            "answer": result.get("answer", ""),
            "source_documents": result.get("source_documents", []),
        }

    def clear_session(self, session_id: str) -> None:
        """Clear chain memory for a session (e.g., on timeout)."""
        if session_id in self._session_chains:
            del self._session_chains[session_id]
