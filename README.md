 Text Summarizer – Explanation
This project is a text summarization tool that uses the BART transformer model (facebook/bart-large-cnn) from Hugging Face.
The script (summarizer.py) summarizes large text files or raw text input by chunking long passages and applying a map-reduce approach.

🔎 Concept
Text summarization is the process of reducing a large body of text into a shorter version while retaining its key information.
There are two main approaches:

Extractive Summarization – selecting important sentences directly from the text.
Abstractive Summarization – generating new sentences that capture the meaning of the original text.
👉 This project uses abstractive summarization with BART, which is pre-trained for summarization tasks.

⚙️ How the Code Works
Input Handling

The script accepts either:
A text file (--input)
A raw string (--text)
If no argument is passed, it defaults to upi.txt.
Preprocessing

clean_text() removes extra spaces, tabs, and unnecessary newlines.
split_into_sentences() breaks the text into sentences.
Chunking

Since BART has a maximum token limit (~1024 tokens), long text is split into smaller chunks (chunk_sentences()).
Overlaps between chunks are added to preserve coherence.
Summarization Pipeline

make_summarizer() loads the Hugging Face pipeline with the model facebook/bart-large-cnn.
Each chunk is summarized individually (summarize_chunk()).
Map-Reduce Summarization

First-pass: Each chunk is summarized separately.
Second-pass: All first-pass summaries are combined and summarized again to produce the final summary.
Styles of Summaries

Short → concise summary (~120–160 words).
Detailed → longer, more detailed summary (~220–320 words).
Bullets → key points formatted as bullet list.
📚 Libraries Used
argparse

For handling command-line arguments (--input, --text, --style).
re (Regular Expressions)

For cleaning text (removing extra spaces, tabs, and formatting sentences).
typing (List)

For type hints (clarity in function signatures).
torch (PyTorch)

Backend for running the summarization model.
Detects whether GPU or CPU is available.
transformers (Hugging Face)

Provides the pipeline API to load the BART model (facebook/bart-large-cnn).
Handles tokenization and generation of summaries.
🧠 Key Idea Behind This Script
Chunk + Summarize (Map-Reduce): Instead of feeding the entire long text (which exceeds model limits), the script breaks it into chunks, summarizes each chunk, and then summarizes the summaries.
Flexible Output: Supports different summary styles for different use cases.
Default File Handling: Automatically uses upi.txt if no input is provided.
✅ Conclusion
This project demonstrates practical abstractive summarization using pretrained transformers.
It shows how to handle large texts, enforce coherent outputs, and provide flexible summary formats using minimal code.
