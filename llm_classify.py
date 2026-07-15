"""
llm_classify.py
Auto-classify knowledge entries and extract keywords using configurable LLM provider.
Supports multiple providers including Gemini, DeepSeek, ZhiPu GLM, Aliyun Qianwen, and Doubao.
"""
import json
import time
import os
from dotenv import load_dotenv
from llm_provider import LLMProvider, call_llm

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(env_path, override=True)

INPUT_FILE = "knowledge_180_marker.json"
OUTPUT_FILE = "knowledge_180_marker_llm.json"
BATCH_SIZE = 5
RATE_LIMIT_DELAY = 4.5

def load_and_process():
    global data, entries
    data = json.load(open(INPUT_FILE, 'r', encoding='utf-8'))
    entries = data['entries']
    print(f"Loaded {len(entries)} entries from {INPUT_FILE}")

PROMPT_TEMPLATE_HEADER = """You are a document classification assistant. For each entry below, determine:
1. type: one of [sop_step, faq, manual_section, general]
   - sop_step: step-by-step instructions or procedures
   - faq: questions and answers
   - manual_section: chapter/section headings or overviews
   - general: everything else (descriptions, specifications, notes)
2. keywords: 3-5 most important keywords from the content

Respond in JSON array format like:
[{"index": 0, "type": "sop_step", "keywords": ["router", "setup", "cable"]}]

Here are the entries:

"""

PROMPT_TEMPLATE_FOOTER = "\n\nRespond ONLY with the JSON array, no other text."

def build_entries_text(batch):
    lines = []
    for i, entry in enumerate(batch):
        title = entry.get('title', '')[:100]
        content = entry.get('content', '')[:300]
        lines.append(f"Entry {i}:\n  Title: {title}\n  Content: {content}\n")
    return "\n".join(lines)

def parse_llm_response(text):
    if text is None:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        elif "```" in text:
            text = text[:text.rfind("```")]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

def process_entries_from_file():
    data = json.load(open(INPUT_FILE, 'r', encoding='utf-8'))
    entries = data['entries']
    print(f"Loaded {len(entries)} entries from {INPUT_FILE}")
    
    total = len(entries)
    processed = 0
    failed = 0
    changes = 0
    from llm_config import get_current_provider, get_current_provider_name
    provider = get_current_provider()
    
    print(f"\nProcessing {total} entries in batches of {BATCH_SIZE} using {get_current_provider_name()}...")
    print(f"Estimated time: ~{(total / BATCH_SIZE) * RATE_LIMIT_DELAY / 60:.1f} minutes\n")
    
    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch = entries[batch_start:batch_end]
        
        entries_text = build_entries_text(batch)
        prompt = PROMPT_TEMPLATE_HEADER + entries_text + PROMPT_TEMPLATE_FOOTER
        
        try:
            text = call_llm(prompt, provider=provider, max_tokens=4096, response_format="json")
            results = parse_llm_response(text)
            
            if results:
                for r in results:
                    idx = batch_start + r['index']
                    if idx < total:
                        old_type = entries[idx].get('type', '')
                        new_type = r.get('type', old_type)
                        if old_type != new_type:
                            changes += 1
                        entries[idx]['llm_type'] = new_type
                        entries[idx]['llm_keywords'] = r.get('keywords', [])
                processed += len(batch)
            else:
                print(f"  [WARN] Batch {batch_start}-{batch_end}: failed to parse response")
                failed += len(batch)
                
        except Exception as e:
            print(f"  [ERROR] Batch {batch_start}-{batch_end}: {str(e)[:80]}")
            failed += len(batch)
        
        pct = (batch_end / total) * 100
        print(f"  [{pct:5.1f}%] Processed {batch_end}/{total} entries | Changes: {changes} | Failed: {failed}")
        
        if batch_end < total:
            time.sleep(RATE_LIMIT_DELAY)
    
    data['entries'] = entries
    data['llm_classification'] = {
        "model": provider,
        "total_processed": processed,
        "total_failed": failed,
        "type_changes": changes,
    }
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Done! Results saved to: {OUTPUT_FILE}")
    print(f"  Processed: {processed}/{total}")
    print(f"  Failed: {failed}")
    print(f"  Type changes (rule vs LLM): {changes}")
    
    print(f"\n{'='*50}")
    print("Type distribution comparison (Rule vs LLM):")
    rule_tc = {}
    llm_tc = {}
    for e in entries:
        rule_tc[e.get('type', '?')] = rule_tc.get(e.get('type', '?'), 0) + 1
        llm_tc[e.get('llm_type', '?')] = llm_tc.get(e.get('llm_type', '?'), 0) + 1
    
    print(f"  {'Type':<20} {'Rule-based':<12} {'LLM':<12}")
    print(f"  {'-'*44}")
    for t in sorted(set(list(rule_tc.keys()) + list(llm_tc.keys()))):
        print(f"  {t:<20} {rule_tc.get(t, 0):<12} {llm_tc.get(t, 0):<12}")

def classify_with_llm(title, content, provider=None):
    prompt = f"""You are a document classification assistant. Analyze the following entry:

Title: {title[:100]}
Content: {content[:300]}

Determine:
1. type: one of [sop_step, faq, manual_section, general]
   - sop_step: step-by-step instructions or procedures
   - faq: questions and answers
   - manual_section: chapter/section headings or overviews
   - general: everything else (descriptions, specifications, notes)
2. keywords: 3-5 most important keywords from the content

Respond ONLY with a JSON object like:
{{"type": "sop_step", "keywords": ["keyword1", "keyword2"], "confidence": 0.85}}
"""
    
    try:
        text = call_llm(prompt, max_tokens=1024, response_format="json")
        parsed = parse_llm_response(text)
        
        if parsed and isinstance(parsed, dict):
            return {
                'type': parsed.get('type', 'general'),
                'keywords': parsed.get('keywords', []),
                'confidence': parsed.get('confidence', 0.7)
            }
        else:
            title_lower = title.lower()
            content_lower = content.lower()
            
            if any(q in title_lower for q in ['how', 'what', 'why', 'when', 'where', 'can', 'is', 'do']):
                return {'type': 'faq', 'keywords': [], 'confidence': 0.6}
            elif any(w in content_lower for w in ['step', 'procedure', 'follow', 'first', 'then', 'next']):
                return {'type': 'sop_step', 'keywords': [], 'confidence': 0.6}
            elif any(w in title_lower for w in ['chapter', 'section', 'introduction', 'overview']):
                return {'type': 'manual_section', 'keywords': [], 'confidence': 0.6}
            else:
                return {'type': 'general', 'keywords': [], 'confidence': 0.5}
                
    except Exception as e:
        print(f"LLM classification error: {e}")
        return {'type': 'general', 'keywords': [], 'confidence': 0.3}

def classify_with_gemini(title, content):
    return classify_with_llm(title, content, provider="gemini")

def classify_batch_with_llm(entries):
    if not entries:
        return []
    
    lines = []
    for i, entry in enumerate(entries):
        title = entry.get('title', '')[:100]
        content = entry.get('content', '')[:300]
        lines.append(f"Entry {i}:\n  Title: {title}\n  Content: {content}\n")
    
    entries_text = "\n".join(lines)
    
    prompt = f"""You are a document classification assistant. Analyze the following {len(entries)} entries and classify each one.

{entries_text}

For each entry, determine:
1. type: one of [sop_step, faq, manual_section, general]
   - sop_step: step-by-step instructions or procedures
   - faq: questions and answers
   - manual_section: chapter/section headings or overviews
   - general: everything else (descriptions, specifications, notes)
2. keywords: 3-5 most important keywords
3. confidence: 0-1 score of how confident you are

Respond ONLY with a JSON array like:
[
  {{"index": 0, "type": "sop_step", "keywords": ["keyword1"], "confidence": 0.85}},
  {{"index": 1, "type": "faq", "keywords": ["keyword2"], "confidence": 0.9}}
]
"""
    
    try:
        text = call_llm(prompt, max_tokens=4096, response_format="json")
        parsed = parse_llm_response(text)
        
        if parsed and isinstance(parsed, list):
            results = [{} for _ in entries]
            for r in parsed:
                idx = r.get('index', -1)
                if 0 <= idx < len(entries):
                    results[idx] = {
                        'type': r.get('type', 'general'),
                        'keywords': r.get('keywords', []),
                        'confidence': r.get('confidence', 0.7)
                    }
            for i in range(len(results)):
                if not results[i]:
                    results[i] = {'type': 'general', 'keywords': [], 'confidence': 0.5}
            return results
        else:
            return [classify_with_llm(e.get('title', ''), e.get('content', '')) for e in entries]
            
    except Exception as e:
        print(f"LLM batch classification error: {e}")
        return [classify_with_llm(e.get('title', ''), e.get('content', '')) for e in entries]

def classify_batch_with_gemini(entries):
    return classify_batch_with_llm(entries)

if __name__ == "__main__":
    process_entries_from_file()