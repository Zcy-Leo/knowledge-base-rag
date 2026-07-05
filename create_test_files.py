import fitz
import os

output_dir = "robustness_test_files"
os.makedirs(output_dir, exist_ok=True)

def create_complex_table_pdf():
    doc = fitz.open()
    page = doc.new_page(width=600, height=800)
    
    table_data = [
        ["", "Q1", "Q2", "Q3", "Q4", "Year Total"],
        ["Product A", "100", "150", "200", "250", "700"],
        ["Product B", "80", "120", "160", "200", "560"],
        ["Product C", "50", "75", "100", "125", "350"],
        ["All Products", "", "", "", "", "1610"],
    ]
    
    y = 100
    row_height = 30
    col_widths = [80, 70, 70, 70, 70, 100]
    
    for i, row in enumerate(table_data):
        x = 50
        for j, cell in enumerate(row):
            rect = fitz.Rect(x, y, x + col_widths[j], y + row_height)
            page.draw_rect(rect, color=(0,0,0), width=0.5)
            
            if i == 0 or j == 0 or i == 4:
                page.insert_text((x + 5, y + 18), cell, fontsize=10)
            else:
                page.insert_text((x + 5, y + 18), cell, fontsize=9)
            
            x += col_widths[j]
        y += row_height
    
    doc.save(os.path.join(output_dir, "complex_table.pdf"))
    doc.close()
    print(f"Created: {output_dir}/complex_table.pdf")

def create_merged_cell_pdf():
    doc = fitz.open()
    page = doc.new_page(width=600, height=800)
    
    y = 100
    row_height = 30
    
    col_widths = [100, 150, 150, 150]
    
    headers = ["Category", "Region A", "Region B", "Region C"]
    x = 50
    for j, h in enumerate(headers):
        rect = fitz.Rect(x, y, x + col_widths[j], y + row_height)
        page.draw_rect(rect, color=(0,0,0), width=0.5)
        page.insert_text((x + 5, y + 18), h, fontsize=10)
        x += col_widths[j]
    y += row_height
    
    categories = ["Hardware", "Software", "Services"]
    for cat in categories:
        x = 50
        
        rect = fitz.Rect(x, y, x + col_widths[0], y + row_height * 3)
        page.draw_rect(rect, color=(0,0,0), width=0.5)
        page.insert_text((x + 5, y + row_height * 1.5), cat, fontsize=10)
        
        x += col_widths[0]
        for j in range(1, 4):
            for k in range(3):
                rect = fitz.Rect(x, y + k * row_height, x + col_widths[j], y + (k + 1) * row_height)
                page.draw_rect(rect, color=(0,0,0), width=0.5)
                value = str((categories.index(cat) + 1) * (j + 1) * (k + 1) * 10)
                page.insert_text((x + 5, y + k * row_height + 18), value, fontsize=9)
            x += col_widths[j]
        y += row_height * 3
    
    doc.save(os.path.join(output_dir, "merged_cell_table.pdf"))
    doc.close()
    print(f"Created: {output_dir}/merged_cell_table.pdf")

def create_corrupted_pdf():
    corrupted_content = b"%PDF-1.4\n%Corrupted\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids []\n>>\nendobj\nxref\n0 3\n0000000000 65535 f \n0000000010 00000 n \n0000000079 00000 n \ntrailer\n<<\n/Size 3\n/Root 1 0 R\n>>\nstartxref\n164\n%%EOF"
    
    with open(os.path.join(output_dir, "corrupted.pdf"), "wb") as f:
        f.write(corrupted_content)
    print(f"Created: {output_dir}/corrupted.pdf")

def create_large_text_file():
    content = "This is a test line. " * 100000
    with open(os.path.join(output_dir, "large_file.txt"), "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Created: {output_dir}/large_file.txt")

create_complex_table_pdf()
create_merged_cell_pdf()
create_corrupted_pdf()
create_large_text_file()

print("\nTest files created successfully!")
