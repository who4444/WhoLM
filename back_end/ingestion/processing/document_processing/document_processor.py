import fitz
import docx  
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap
                                                       )
    def process_file(self, path):
        raw_text = self.file_router(path)
        
        # If no text was extracted, return empty list
        if not raw_text:
            return {"text_chunks": [],
                    "success": False}

        # Apply the splitter to the extracted text
        # split_text returns a list of strings
        chunks = self.splitter.split_text(raw_text)
        return {"text_chunks": chunks,
                "success": True}

    def file_router(self, path):
        if path.endswith('.pdf'):
            return self.load_pdf(path)
        elif path.endswith('.docx'):
            return self.load_docx(path)
        else:
            # Default to text loader
            return self.load_txt(path)

    def load_pdf(self, path):
        pdf = fitz.open(path)
        all_chunks = []

        for page_index, page in enumerate(pdf):
            text = page.get_text()
            all_chunks.append(text)

        return "\n\n".join(all_chunks)
    

    def load_docx(self, path):
        doc = docx.Document(path)
        # Extract all paragraphs and join them
        full_text = "\n".join([para.text for para in doc.paragraphs if para.text])
        return full_text

    def load_txt(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading txt file: {e}")
            return ""

