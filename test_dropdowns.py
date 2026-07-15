import os
import sys
sys.path.insert(0, '.')

import streamlit as st

from app_v2 import get_all_companies, get_all_topics, get_metadata_counts, DB_PATH

print(f"[TEST] DB_PATH: {DB_PATH}")
print(f"[TEST] DB exists: {os.path.exists(DB_PATH)}")

st.title("Test Dropdowns")

# Test 1: Simple dropdown
st.subheader("Test 1: Simple Dropdown")
companies = get_all_companies()
st.write(f"Companies: {companies}")
selected_company = st.selectbox("Select Company", ["--- Select ---"] + companies)
st.write(f"Selected: {selected_company}")

# Test 2: Dropdown with counts
st.subheader("Test 2: Dropdown with counts")
company_counts, topic_counts = get_metadata_counts()
company_options = ["--- Select Company ---"]
for c in companies:
    count = company_counts.get(c, 0)
    company_options.append(f"{c} ({count})")
company_options.append("+ Add New Company")
st.write(f"Company options: {company_options}")
selected_company_raw = st.selectbox("Select Company (with counts)", company_options)
st.write(f"Selected raw: {selected_company_raw}")

# Test 3: Topics
st.subheader("Test 3: Topics")
topics = get_all_topics()
st.write(f"Topics: {topics}")
topic_options = ["--- Select Topic ---"]
for t in topics:
    count = topic_counts.get(t, 0)
    topic_options.append(f"{t} ({count})")
st.write(f"Topic options: {topic_options}")
selected_topic_raw = st.selectbox("Select Topic", topic_options)
st.write(f"Selected topic: {selected_topic_raw}")
