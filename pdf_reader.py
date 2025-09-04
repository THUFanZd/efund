import pdfplumber

with pdfplumber.open(r"C:\Users\lzx\Desktop\大四暑\易方达杯\人民银行文本\2025-2.pdf") as pdf:
    count = 0
    for page in pdf.pages:
        count += 1
        if count == 8:
            print(page.extract_text())
            print(page.extract_tables())
            break

