from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
import textwrap
import asyncio

app = FastAPI()

# Initialize the Ollama Llama3 model
llama3 = Ollama(model='llama3')


# Define a request body model
class TextRequest(BaseModel):
    text: str


def chunk_text(text, chunk_size=1000):
    """Split text into chunks of a given size."""
    return textwrap.wrap(text, chunk_size)


def summarize_chunk(chunk):
    """Summarize a single chunk of text."""
    prompt_template = PromptTemplate.from_template(
        "Summarize the following code:\n\n{text}\n\nSummary:"
    )
    prompt = prompt_template.format(text=chunk)

    # Get summary from Llama3 model
    summary = llama3.invoke(prompt)
    return summary


@app.post("/summarize")
def summarize(request: TextRequest):
    text = request.text
    chunk_size = 1000  # Adjust chunk size as needed
    chunks = chunk_text(text, chunk_size)

    # Summarize each chunk
    summaries = [summarize_chunk(chunk) for chunk in chunks]

    # Combine summaries
    combined_summary_prompt = PromptTemplate.from_template(
        "Combine the following summaries into a concise summary:\n\n{summaries}\n\nFinal Summary:"
    )
    combined_summary_prompt_text = combined_summary_prompt.format(summaries="\n".join(summaries))

    final_summary = llama3.invoke(combined_summary_prompt_text)

    return {"summary": final_summary}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)