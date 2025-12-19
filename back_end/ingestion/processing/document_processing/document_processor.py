import fitz
from llama_index.core.node_parser import SentenceSplitter
import docx  

class DocumentProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    def process_file(self, path):
        raw_text = self.file_router(path)
        
        # If no text was extracted, return empty list
        if not raw_text:
            return []

        # Apply the splitter to the extracted text
        # split_text returns a list of strings
        chunks = self.splitter.split_text(raw_text)
        return chunks

    def file_router(self, path):
        if path.endswith('.pdf'):
            return self.load_pdf(path)
        elif path.endswith('.docx'):
            return self.load_docx(path)
        elif path.endswith('.csv'):
            return self.load_csv(path)
        else:
            # Default to text loader
            return self.load_txt(path)

    def load_pdf(self, path):
        pdf = fitz.open(path)
        all_chunks = []

        for page_index, page in enumerate(pdf):
            # 1. Extract Text
            text = page.get_text()
            all_chunks.append(text)

            # 2. Extract Images
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = pdf.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Save Image to Disk 
                image_filename = f"{self.output_img_dir}/page_{page_index}_img_{img_index}.png"
                with open(image_filename, "wb") as f:
                    f.write(image_bytes)

                try:
                    # Convert bytes to base64 string for API 
                    description = self.generate_image_summary(image_bytes) 
                    image_chunk = f"Image Context: {description}"
                    all_chunks.append(image_chunk)
                    
                except Exception as e:
                    print(f"Failed to process image: {e}")

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

