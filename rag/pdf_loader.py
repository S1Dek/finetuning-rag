import fitz  # PyMuPDF

def extract_pdf(path):
    doc = fitz.open(path)

    text = []
    images = []

    for page in doc:
        text.append(page.get_text())

        for img in page.get_images(full=True):
            xref = img[0]
            base = doc.extract_image(xref)
            images.append(base["image"])

    return "\n".join(text), images