"""
表格提取最终对比：Marker vs PyMuPDF find_tables()
直接读取已有的marker结果，与PyMuPDF对比
"""

import json
import fitz

# 测试文件
PDF_PATH = "e:\\code\\c\\Knowledge Base Automation\\organized_documents\\PDF_Documents\\all-number-table.pdf"
MARKER_JSON = "e:\\code\\c\\Knowledge Base Automation\\pdf_results\\knowledge_all-number-table (4)_marker.json"

print("=" * 70)
print("表格提取最终对比：Marker vs PyMuPDF find_tables()")
print("=" * 70)
print()

# ============================================
# 方法1: Marker结果（已解析好）
# ============================================
print("【方法1】Marker 结果")
print("-" * 50)

with open(MARKER_JSON, 'r', encoding='utf-8') as f:
    marker_data = json.load(f)

print(f"文件: {marker_data['source_file']}")
print(f"总条目数: {marker_data['total_entries']}")

for entry in marker_data['entries']:
    if entry['type'] == 'table_data':
        print(f"\n检测到表格条目:")
        print(f"  type: {entry['type']}")
        print(f"  title: {entry['title'][:50]}...")
        print(f"  source_page: {entry['source_page']}")
        print(f"\n表格内容 (Markdown格式):")
        print(entry['content'])
        print()

# ============================================
# 方法2: PyMuPDF find_tables()
# ============================================
print("【方法2】PyMuPDF find_tables()")
print("-" * 50)

doc = fitz.open(PDF_PATH)
page = doc[0]
tables = page.find_tables()

if tables.tables:
    for i, table in enumerate(tables.tables):
        data = table.extract()
        print(f"\n检测到表格 {i+1}:")
        print(f"  行数: {len(data)}, 列数: {len(data[0]) if data else 0}")
        print(f"  边界框: {table.bbox}")
        print(f"\n表格内容 (二维数组):")
        for row in data:
            print(f"  {row}")
        print()
doc.close()

# ============================================
# 对比总结
# ============================================
print("=" * 70)
print("对比总结")
print("=" * 70)

print("""
| 维度 | Marker | PyMuPDF find_tables() |
|------|--------|----------------------|
| 输出格式 | Markdown表格 | 二维数组 |
| 自动分类 | ✓ 自动标记为 table_data | ✗ 无分类 |
| 表格标题 | ✓ 保留表格标题 | ✗ 无标题 |
| 源页码 | ✓ 记录source_page | ✗ 需手动记录 |
| 处理速度 | 慢(需AI模型) | 快(纯文本分析) |
| RAG友好 | ✓ Markdown可向量化 | ~ 需转字符串 |
| 数据分析 | ~ 需解析Markdown | ✓ 直接DataFrame |
""")

print("【建议】")
print()
print("场景1: 构建知识库/RAG系统 → 用Marker")
print("  - 表格自动分类为 table_data")
print("  - Markdown格式可直接向量化")
print("  - 保留表格标题和上下文")
print("  - 你已经在用marker了，表格效果很好")
print()
print("场景2: 纯数据提取/数据分析 → 用PyMuPDF find_tables()")
print("  - 二维数组直接转DataFrame")
print("  - 速度快，适合批量处理")
print("  - 但需要额外处理元数据")
print()
print("【最终结论】")
print("既然你的目标是构建企业知识库，且marker已经:")
print("  ✓ 自动识别表格类型 (table_data)")
print("  ✓ 保留完整Markdown格式")
print("  ✓ 记录source_page等元数据")
print()
print("建议：继续用Marker，不要换PyMuPDF find_tables()")
print("PyMuPDF find_tables()更适合数据提取场景，不适合RAG")