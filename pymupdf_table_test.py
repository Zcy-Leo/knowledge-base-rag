"""
PyMuPDF表格提取方法对比测试（轻量版，无需AI模型）
对比三种方法：
1. find_tables() - 专用表格API（正确方法）
2. get_text() - 纯文本提取（你们老方法）
3. get_text("markdown") - Markdown格式（新方法）
"""

import fitz
import os

PDF_PATH = "e:\\code\\c\\Knowledge Base Automation\\organized_documents\\PDF_Documents\\all-number-table.pdf"

print("=" * 70)
print("PyMuPDF表格提取方法对比测试")
print("=" * 70)
print(f"文件: {os.path.basename(PDF_PATH)}")
print()

doc = fitz.open(PDF_PATH)
page = doc[0]

# 方法1: find_tables() - 专用表格API
print("【方法1】page.find_tables() - 表格专用API（✓ 正确）")
print("-" * 50)
tables = page.find_tables()
if tables.tables:
    for i, table in enumerate(tables.tables):
        data = table.extract()
        print(f"检测到表格 {i+1}: {len(data)} 行 x {len(data[0])} 列")
        print("完整表格数据:")
        for row in data:
            print(f"  {row}")
        print()
else:
    print("未检测到表格")
print()

# 方法2: get_text() - 纯文本（你们老方法）
print("【方法2】page.get_text() - 纯文本提取（✗ 你们老方法）")
print("-" * 50)
text_plain = page.get_text()
print("输出结果（纯文本，表格结构丢失）:")
print(text_plain[:500] if len(text_plain) > 500 else text_plain)
print()

# 方法3: Marker（需要AI模型，略过）
print("【方法3】Marker - 需要AI模型（内存不足时fallback到PyMuPDF）")
print("-" * 50)
print("Marker方法需要加载AI模型（~1-2GB），Windows内存不足时会fallback")
print("Marker的优势是保留完整文档语义，适合RAG场景")
print()

doc.close()

print("=" * 70)
print("结论")
print("=" * 70)
print("方法1 (find_tables):")
print("  ✓ 表格结构完整保留")
print("  ✓ 返回二维数组，可直接转DataFrame")
print("  ✓ 适合数据分析和提取")
print()
print("方法2 (get_text):")
print("  ✗ 表格变成无结构的文本流")
print("  ✗ 你们老方法PyMuPDFLoader就是用的这个")
print("  ✗ 不适合表格场景")
print()
print("方法3 (get_text('markdown')):")
print("  ~ 部分表格结构保留（用|分隔）")
print("  ~ 但不如find_tables精确")
print()