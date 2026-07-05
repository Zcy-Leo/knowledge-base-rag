"""
表格提取对比测试：PyMuPDF vs Marker
测试文件：layout-parser-paper-with-table.pdf
"""

import fitz  # PyMuPDF
import time
import os

# 选择测试文件
TEST_FILES = [
    "e:\\code\\c\\Knowledge Base Automation\\organized_documents\\PDF_Documents\\all-number-table.pdf",
    "e:\\code\\c\\Knowledge Base Automation\\organized_documents\\PDF_Documents\\single_table.pdf",
]

PDF_PATH = TEST_FILES[0]  # 默认使用第一个

print("=" * 60)
print("表格提取对比测试：PyMuPDF find_tables() vs Marker")
print("=" * 60)
print(f"测试文件: {os.path.basename(PDF_PATH)}")
print()

# ============================================
# 方法1: PyMuPDF find_tables()
# ============================================
print("【方法1】PyMuPDF find_tables() 表格专用API")
print("-" * 40)

start_time = time.time()

doc = fitz.open(PDF_PATH)
total_tables = 0
pymupdf_results = []

for page_num in range(len(doc)):
    page = doc[page_num]
    tables = page.find_tables()
    
    if tables.tables:
        for i, table in enumerate(tables.tables):
            total_tables += 1
            data = table.extract()
            bbox = table.bbox
            
            pymupdf_results.append({
                "page": page_num + 1,
                "table_index": i + 1,
                "bbox": bbox,
                "rows": len(data),
                "cols": len(data[0]) if data else 0,
                "data": data
            })
            
            print(f"Page {page_num + 1}, Table {i + 1}:")
            print(f"  位置: ({bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f})")
            print(f"  行数: {len(data)}, 列数: {len(data[0]) if data else 0}")
            print(f"  内容预览 (前3行):")
            for row_idx, row in enumerate(data[:3]):
                print(f"    Row {row_idx + 1}: {row}")
            print()

doc.close()
pymupdf_time = time.time() - start_time

print(f"PyMuPDF 总耗时: {pymupdf_time:.2f} 秒")
print(f"PyMuPDF 检测到表格数: {total_tables}")
print()

# ============================================
# 方法2: Marker
# ============================================
print("【方法2】Marker 本地推理")
print("-" * 40)

start_time = time.time()

from marker_extractor import parse_pdf_with_marker

md_text = parse_pdf_with_marker(PDF_PATH, max_pages=3)  # 只解析前3页节省时间
marker_time = time.time() - start_time

# 从Markdown中提取表格（寻找 | 分隔的表格行）
print("Marker 输出 (Markdown格式):")
print()

# 提取表格部分
lines = md_text.split('\n')
table_lines = []
in_table = False

for line in lines:
    if '|' in line and line.strip().startswith('|'):
        in_table = True
        table_lines.append(line)
    elif in_table and line.strip() == '':
        # 表格结束
        if table_lines:
            print("检测到Markdown表格:")
            for tl in table_lines[:10]:  # 只显示前10行
                print(tl)
            print(f"  (共 {len(table_lines)} 行)")
            print()
            table_lines = []
            in_table = False
    elif in_table:
        table_lines.append(line)

# 如果还有剩余表格
if table_lines:
    print("检测到Markdown表格:")
    for tl in table_lines[:10]:
        print(tl)
    print(f"  (共 {len(table_lines)} 行)")
    print()

print(f"Marker 总耗时: {marker_time:.2f} 秒")
print()

# ============================================
# 对比总结
# ============================================
print("=" * 60)
print("对比总结")
print("=" * 60)

print(f"| 指标 | PyMuPDF | Marker |")
print(f"|------|---------|--------|")
print(f"| 耗时 | {pymupdf_time:.2f}s | {marker_time:.2f}s |")
print(f"| 检测表格数 | {total_tables} | Markdown表格行数统计 |")
print(f"| 输出格式 | 二维数组 | Markdown |")
print(f"| 表格结构 | 结构化数据 | 格式化文本 |")
print()

print("结论：")
print("- PyMuPDF: 快速、结构化、适合数据提取到DataFrame")
print("- Marker: 较慢但保留完整文档语义，适合RAG场景")
print("- 如果只需要表格数据，PyMuPDF更高效")
print("- 如果需要保留文档上下文，Marker更合适")