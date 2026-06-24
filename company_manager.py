"""
company_manager.py
============================
Manages company list for knowledge base system.
Supports adding, listing, and persisting companies.
"""

import json
import os

COMPANY_FILE = "./companies.json"

def load_companies() -> list:
    """Load companies from JSON file."""
    if os.path.exists(COMPANY_FILE):
        try:
            with open(COMPANY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("companies", [])
        except Exception:
            return []
    return []

def save_companies(companies: list) -> None:
    """Save companies to JSON file."""
    data = {"companies": sorted(list(set(companies)))}
    with open(COMPANY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def add_company(name: str) -> bool:
    """Add a new company if it doesn't exist."""
    name = name.strip()
    if not name:
        return False
    
    companies = load_companies()
    if name not in companies:
        companies.append(name)
        save_companies(companies)
        return True
    return False

def get_companies() -> list:
    """Get all companies sorted alphabetically, with NA always available."""
    companies = sorted(load_companies())
    if "NA" not in companies:
        companies.insert(0, "NA")
    return companies

def remove_company(name: str) -> bool:
    """Remove a company from the list."""
    companies = load_companies()
    if name in companies:
        companies.remove(name)
        save_companies(companies)
        return True
    return False

def has_company(name: str) -> bool:
    """Check if a company exists."""
    return name.strip() in load_companies()

if __name__ == "__main__":
    # Test the manager
    print("Testing Company Manager...")
    
    # Add some test companies
    add_company("Samsung")
    add_company("HP")
    add_company("Apple")
    
    # Get all companies
    companies = get_companies()
    print(f"Companies: {companies}")
    
    # Check if company exists
    print(f"Has Samsung: {has_company('Samsung')}")
    print(f"Has Sony: {has_company('Sony')}")
    
    # Add a new company
    added = add_company("Sony")
    print(f"Added Sony: {added}")
    
    # Try adding duplicate
    added = add_company("Samsung")
    print(f"Added Samsung again: {added}")
    
    # Remove a company
    removed = remove_company("Apple")
    print(f"Removed Apple: {removed}")
    
    # Final list
    companies = get_companies()
    print(f"Final companies: {companies}")
