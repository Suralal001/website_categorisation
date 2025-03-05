import os
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from langchain_openai import ChatOpenAI
from bs4 import BeautifulSoup
import requests
import time
import re
import asyncio
import nest_asyncio
import tiktoken  # Add tiktoken for token counting

from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig

nest_asyncio.apply()

# Excluded Categories
EXCLUDED_CATEGORIES = {"About Us", "SEO", "Contact Us", "Disclaimer", "Privacy Policy", 
                       "Terms and Conditions", "X", "Instagram", "Pinterest", "Home", 
                       "Business", "Entertainment", "Technology", "Pick Up Lines"}

# Categories Related to News
NEWS_RELATED_CATEGORIES = {"Automobile", "Education", "Fashion", "Finance", "Food", "Game", 
                           "Health", "Home Improvement", "Law", "Lifestyle", "News", "Others", 
                           "Pet", "Real Estate", "SEO", "Tech Lines", "Travel"}

# Allowed Categories
VALID_CATEGORIES = {"Finance", "RealEstate", "Legal", "Marketing", "Ecommerce", "Startups", 
                    "Cryptocurrency", "Travel", "Food", "Health", "Fitness", "Beauty", 
                    "Fashion", "Home", "Parenting", "Sustainability", "Automotive", "Tech", 
                    "AI", "Cybersecurity", "Gaming", "Science", "Education", "Productivity", 
                    "History", "Psychology", "News", "Music", "Film", "Books", "Art", 
                    "Adult", "Gambling", "Cannabis", "Medical"}

CATEGORY_MAPPING = {
    "Business": ["Entrepreneurship", "Startups", "Marketing", "Ecommerce", "Finance", "Sales", "Leadership", "Workplace"],
    "Technology": ["Artificial Intelligence", "Cybersecurity", "Crypto", "Software", "Gaming", "Gadgets", "Big Data", "Cloud Computing", "Automation", "Virtual Reality", "Biotech"],
    "Science": ["Space", "Biology", "Physics", "Engineering", "Medicine", "Neuroscience", "Psychology", "History", "Archaeology", "Education"],
    "Travel": ["Adventure Travel", "Luxury Travel", "Backpacking", "Cultural Travel", "Food Travel", "Sustainable Travel", "Road Trips", "Van Life", "Surf Travel", "Ski Travel"],
    "Lifestyle": ["Food", "Fitness", "Beauty", "Fashion", "Parenting", "Home", "Self Help", "Breweries & Brewpubs", "Cideries & Meaderies", "Spirits", "Beverages"],
    "Health": ["Medical Services", "Healthcare", "Mental Health", "Wellness", "Nutrition", "Telemedicine", "Pharmacy", "Insurance", "Doctors", "Hospitals", "Clinics"],
    "Automotive": ["Electric Vehicles", "Car Culture", "Motorcycles", "Aviation", "Public Transport"],
    "Legal": ["Business Law", "Contracts", "Intellectual Property", "Compliance"],
    "Entertainment": ["Music", "Film", "Books", "Art", "Theater"],
    "News": ["Current Events", "Politics", "Global Issues", "Social Trends"],
    "Adult": ["Gambling", "Cannabis", "Psychedelics", "Cannabis Products", "Cannabis Industry", "Cannabis Reviews"]
}

# OpenAI API Key
OPENAI_API_KEY = ""

# Authenticate Google Sheets
def authenticate_google_sheets():
    scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("google_sheets_credentials.json", scope)
    return gspread.authorize(creds)

# 🕷️ **Async Web Crawler**
async def simple_crawl(website_url):
    """Asynchronously crawls the website and extracts categories from its menu"""
    crawler_run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
    if not website_url.startswith(("http://", "https://")):
        website_url = "https://" + website_url  

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=website_url, config=crawler_run_config)

        extracted_text = result.markdown_v2.raw_markdown if result and hasattr(result, 'markdown_v2') else ""

        # Extract menu items
        menu_items = re.findall(r"\* \[(.*?)\]", extracted_text)
        menu_items = [item.strip() for item in menu_items if item.strip() not in EXCLUDED_CATEGORIES]

        # Initialize variables
        primary_categories = set()
        secondary_categories_list = []
        
        # Category mapping for common variations
        category_variations = {
            "Business": ["Business", "finance", "Real Estate", "SEO"],
            "Technology": ["Technology", "Tech Lines", "App"],
            "Entertainment": ["Entertainment", "Game"],
            "Lifestyle": ["Fashion", "Food", "Fitness", "Beauty", "Home", "Home Improvement", "Pet"],
            "Health": ["Health", "Medical", "Healthcare", "Wellness", "Doctor", "Clinic", "Hospital", "Pharmacy", 
                      "Medicine", "Mental Health", "Therapy", "Patient", "Care", "Treatment"],
            "Legal": ["LAW", "law", "Legal"],
            "News": ["News"],
            "Travel": ["Travel"],
            "Automotive": ["Automobile"]
        }
        
        # Check for health-related terms in the website content
        health_terms = ["health", "medical", "doctor", "clinic", "hospital", "patient", "care", "wellness", 
                        "treatment", "therapy", "medicine", "healthcare", "physician", "telemedicine",
                        "appointment", "diagnosis", "prescription", "specialist", "nurse", "emergency"]
        
        # If health terms are found in the content, add Health to primary categories
        health_term_count = sum(1 for term in health_terms if term in extracted_text.lower())
        
        # If multiple health terms are found, it's likely a health website
        if health_term_count >= 3:
            primary_categories.add("Health")
            # Add relevant secondary categories based on specific health terms
            if "mental" in extracted_text.lower():
                secondary_categories_list.append("Mental Health")
            if "telemedicine" in extracted_text.lower() or "virtual" in extracted_text.lower():
                secondary_categories_list.append("Telemedicine")
            if "insurance" in extracted_text.lower():
                secondary_categories_list.append("Insurance")
            if "wellness" in extracted_text.lower():
                secondary_categories_list.append("Wellness")
            if "nutrition" in extracted_text.lower() or "diet" in extracted_text.lower():
                secondary_categories_list.append("Nutrition")
            
            # If we found health terms but also business/lifestyle terms, prioritize health
            if "Business" in primary_categories or "Lifestyle" in primary_categories:
                primary_categories.discard("Business")
                primary_categories.discard("Lifestyle")

        # Map found categories to our primary categories
        for menu_item in menu_items:
            for primary_cat, variations in category_variations.items():
                if any(variation.lower() in menu_item.lower() for variation in variations):
                    primary_categories.add(primary_cat)
                    # Add the menu item as a secondary category
                    if not any(variation.lower() == menu_item.lower() for variation in variations[:1]):
                        secondary_categories_list.append(menu_item)

        # Convert primary categories to string
        if len(primary_categories) > 5:
            primary_category = "General"
        else:
            primary_category = ", ".join(sorted(primary_categories)) if primary_categories else "Uncategorized"

        # Handle secondary categories
        if secondary_categories_list:
            secondary_categories = ", ".join(sorted(set(secondary_categories_list)))
        else:
            secondary_categories = "None"

        # Determine if it's niche
        niche = "No" if len(primary_categories) > 1 or primary_category == "General" else "Yes"

    return {
        "primary_category": primary_category,
        "secondary_categories": secondary_categories,
        "niche": niche,
        "status": "success"
    }
    
# 🚀 **Categorization via LLM**
def categorize_website(llm, content):
    """Uses LLM to categorize website content and generate a detailed description."""
    try:
        print(f"Received content for analysis: {content[:100]}...")  # Print first 100 chars of content

        # Check for health-related terms in the content
        health_terms = ["health", "medical", "doctor", "clinic", "hospital", "patient", 
                       "care", "wellness", "treatment", "therapy", "medicine", "healthcare",
                       "physician", "telemedicine", "pharmacy"]
        
        # More specific health context terms that should appear near health terms
        health_context_terms = ["appointment", "diagnosis", "prescription", "specialist", 
                               "symptom", "disease", "patient", "doctor", "medical", "clinical"]
        
        # Check for ecommerce-specific terms
        ecommerce_terms = ["ecommerce", "e-commerce", "online store", "shopify", "woocommerce", 
                          "online business", "digital marketing", "conversion rate", "cart abandonment",
                          "product page", "checkout", "online sales", "online retail"]
        
        # Count health terms and check for health context
        health_term_count = sum(1 for term in health_terms if term in content.lower())
        health_context_count = sum(1 for term in health_context_terms if term in content.lower())
        ecommerce_term_count = sum(1 for term in ecommerce_terms if term in content.lower())
        
        # Only consider it health content if:
        # 1. Multiple health terms are found AND
        # 2. At least one health context term is found AND
        # 3. There aren't significantly more ecommerce terms than health terms
        has_health_content = health_term_count >= 3 and health_context_count >= 1 and health_term_count > ecommerce_term_count
        
        # Simplified prompt to reduce token usage
        prompt = f"""
            Analyze this website content and categorize it:

            **Primary Categories (choose 1-3):**  
            Business, Technology, Science, Travel, Lifestyle, Automotive, Legal, Entertainment, News, Adult, Health  

            **Secondary Categories (choose up to 3 per primary):**  
            Business: Entrepreneurship, Startups, Marketing, Ecommerce, Finance, Sales, Leadership, Workplace
            Technology: Artificial Intelligence, Cybersecurity, Crypto, Software, Gaming, Gadgets, Big Data, Cloud Computing
            Science: Space, Biology, Physics, Engineering, Medicine, Neuroscience, Psychology, History, Education
            Travel: Adventure Travel, Luxury Travel, Backpacking, Cultural Travel, Food Travel, Sustainable Travel
            Lifestyle: Food, Fitness, Beauty, Fashion, Parenting, Home, Self Help, Beverages
            Health: Medical Services, Healthcare, Mental Health, Wellness, Nutrition, Telemedicine, Pharmacy, Insurance
            Automotive: Electric Vehicles, Car Culture, Motorcycles, Aviation, Public Transport
            Legal: Business Law, Contracts, Intellectual Property, Compliance
            Entertainment: Music, Film, Books, Art, Theater
            News: Current Events, Politics, Global Issues, Social Trends
            Adult: Gambling, Cannabis, Psychedelics, Cannabis Products, Cannabis Industry

            **Content:**
            {content}
            
            **Important Notes:**
            1. If the content contains medical services, healthcare information, doctor/patient information, 
               or health-related topics, categorize it primarily as Health rather than Lifestyle or Business.
            2. Only use the Adult category when there is explicit adult content (gambling, cannabis, etc.).
            3. Be precise with categorization - don't add categories unless they are clearly represented in the content.
            4. DO NOT add cannabis-related categories unless the website explicitly focuses on cannabis products or industry.

            **Response Format:**
            PRIMARY: [Category1, Category2]
            SECONDARY: [Category1]: subcategory1, subcategory2; [Category2]: subcategory1
            NICHE: [Yes/No]
            STATUS: [Success/Failed]
            """
            
        # Estimate token count and reduce content if needed
        estimated_tokens = estimate_tokens(prompt)
        print(f"Estimated token count: {estimated_tokens}")
        
        # If over limit, truncate content
        if estimated_tokens > 6000:  # Reduced from 7000 to 6000 to leave more buffer
            # Calculate how many tokens we need to remove
            excess_tokens = estimated_tokens - 6000
            content_tokens = estimate_tokens(content)
            
            # If content is the main contributor to token count
            if content_tokens > excess_tokens:
                # Calculate what percentage of content to keep
                keep_ratio = (content_tokens - excess_tokens) / content_tokens
                words = content.split()
                new_word_count = max(100, int(len(words) * keep_ratio))  # Ensure at least 100 words
                content = ' '.join(words[:new_word_count]) + "... [content truncated]"
                prompt = prompt.replace("{content}", content)
                print(f"Content truncated to approximately {new_word_count} words")
                print(f"New estimated token count: {estimate_tokens(prompt)}")
            else:
                # If we still have too many tokens, use a more aggressive approach
                words = content.split()
                content = ' '.join(words[:250]) + "... [content severely truncated]"
                prompt = prompt.replace("{content}", content)
                print(f"Content severely truncated to 250 words")
                print(f"New estimated token count: {estimate_tokens(prompt)}")

        print("Sending prompt to LLM...")
        response = llm.invoke(prompt)
        raw_text = response.content.strip()
        print(f"Received LLM response: {raw_text[:100]}...")  # Print first 100 chars of response

        # Handle LLM failures
        if any(error_msg in raw_text.lower() for error_msg in ["i cannot browse", "unable to determine", "i'm sorry"]):
            return {
                "primary_category": "Uncategorized", 
                "secondary_categories": "None", 
                "niche": "No", 
                "status": "Failed"
            }

        # Check for cannabis-related content in the entire response
        cannabis_terms = ["cannabis", "marijuana", "cbd", "thc", "hemp", "dispensary"]
        has_cannabis_content = any(term in raw_text.lower() for term in cannabis_terms)

        # Extract and process primary categories
        primary_match = re.search(r"PRIMARY:\s*(.*?)(?:\n|$)", raw_text)
        primary_categories = primary_match.group(1).strip() if primary_match else "Uncategorized"
        
        # Add Adult category only if cannabis content is found AND Adult isn't already included
        # AND the content is specifically about cannabis products/industry (not medical use)
        if has_cannabis_content and "Adult" not in primary_categories and not has_health_content:
            if primary_categories == "Uncategorized":
                primary_categories = "Adult"
            else:
                primary_categories = f"{primary_categories}, Adult"

        # If health terms are found but Health category isn't included, prioritize it
        if has_health_content and "Health" not in primary_categories:
            # Check if this is primarily an ecommerce website
            is_primarily_ecommerce = "Ecommerce" in secondary_categories or ecommerce_term_count >= 5
            
            # For ecommerce websites, add Health as a category but don't replace Business
            if is_primarily_ecommerce and "Business" in primary_categories:
                # Add Health without replacing Business
                if primary_categories == "Business":
                    primary_categories = "Business, Health"
                else:
                    # Add Health if not already in the list
                    primary_categories = f"{primary_categories}, Health" if "Health" not in primary_categories else primary_categories
            else:
                # For non-ecommerce sites, follow the original logic
                if "Business" in primary_categories and "Lifestyle" in primary_categories:
                    primary_categories = primary_categories.replace("Business, Lifestyle", "Health")
                elif "Business" in primary_categories:
                    primary_categories = primary_categories.replace("Business", "Health")
                elif "Lifestyle" in primary_categories:
                    primary_categories = primary_categories.replace("Lifestyle", "Health")
                else:
                    # If neither Business nor Lifestyle is present, add Health
                    if primary_categories == "Uncategorized":
                        primary_categories = "Health"
                    else:
                        primary_categories = f"Health, {primary_categories}"

        # Extract secondary categories with their primary categories
        secondary_section = re.search(r"SECONDARY:\s*(.*?)(?=NICHE:|$)", raw_text, re.DOTALL)
        if secondary_section:
            secondary_text = secondary_section.group(1).strip()
            subcategories = []
            
            # Extract from structured format [Category]: subcats
            structured_matches = re.finditer(r'\[(.*?)\]:\s*(.*?)(?=\n\[|\n\s*$|$)', secondary_text, re.DOTALL)
            for match in structured_matches:
                subcats = [s.strip() for s in match.group(2).split(',')]
                subcategories.extend(subcats)
            
            # If no structured matches found, try to extract as plain list
            if not subcategories:
                subcategories = [s.strip() for s in secondary_text.replace('[', '').replace(']', '').split(',')]
                subcategories = [s for s in subcategories if s and not s.endswith(':')]
            
            # Clean up subcategories
            subcategories = [s for s in subcategories if s and not any(x in s.lower() for x in ['none', 'n/a'])]
            
            if subcategories:
                secondary_categories = ", ".join(sorted(set(subcategories)))
            else:
                secondary_categories = "None"
        else:
            secondary_categories = "None"

        niche_match = re.search(r"NICHE:\s*(.*?)(?:\n|$)", raw_text)
        niche = niche_match.group(1).strip() if niche_match else "No"
        
        # If primary is General, niche should be No
        if primary_categories == "General":
            niche = "No"

        status_match = re.search(r"STATUS:\s*(.*?)(?:\n|$)", raw_text)
        status = status_match.group(1).strip() if status_match else "Failed"

        return {
            "primary_category": primary_categories, 
            "secondary_categories": secondary_categories, 
            "niche": niche, 
            "status": status
        }

    except Exception as e:
        return {
            "primary_category": "Error", 
            "secondary_categories": "None", 
            "niche": "No", 
            "status": f"Error: {str(e)}"
        }

# 🌐 **Website Content Fetcher**
async def fetch_website_content(url):
    """Fetches website content for LLM analysis."""
    try:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            if result and hasattr(result, 'markdown_v2'):
                # Extract content and significantly reduce size to avoid token limit errors
                content = result.markdown_v2.raw_markdown
                
                # Extract the most important parts: headings, first paragraphs, and menu items
                important_content = []
                
                # Get headings (lines starting with #)
                headings = [line for line in content.split('\n') if line.strip().startswith('#')]
                important_content.extend(headings[:5])  # Reduced from 10 to 5 headings
                
                # Get first few paragraphs
                paragraphs = [p for p in content.split('\n\n') if p and not p.strip().startswith('#')]
                important_content.extend(paragraphs[:3])  # Reduced from 5 to 3 paragraphs
                
                # Get menu items
                menu_items = re.findall(r"\* \[(.*?)\]", content)
                if menu_items:
                    important_content.append("Menu items: " + ", ".join(menu_items[:10]))  # Reduced from 15 to 10 menu items
                
                # Check for health-related terms and prioritize those sections
                health_terms = ["health", "medical", "doctor", "clinic", "hospital", "patient", 
                               "care", "wellness", "treatment", "therapy", "medicine", "healthcare"]
                
                health_paragraphs = []
                for p in paragraphs:
                    if any(term in p.lower() for term in health_terms):
                        health_paragraphs.append(p)
                
                # Add health-related paragraphs if found (up to 2)
                if health_paragraphs:
                    important_content.append("Health-related content:")
                    important_content.extend(health_paragraphs[:2])  # Reduced from 3 to 2 health paragraphs
                
                # Combine and limit to 500 words max (about 650 tokens)
                extracted_content = "\n\n".join(important_content)
                words = extracted_content.split()
                if len(words) > 500:  # Reduced from 800 to 500 words
                    extracted_content = ' '.join(words[:500]) + "... [content truncated]"
                
                return extracted_content
            return f"Unable to fetch content for {url}"
            
    except Exception as e:
        return f"Error fetching content: {str(e)}"

# 📊 **Google Sheets Updater**
async def update_google_sheet(force_recategorize_websites=None):
    """Updates missing categories in Google Sheets using batch updates.
    
    Args:
        force_recategorize_websites: List of website URLs to recategorize even if they already have data
    """
    if force_recategorize_websites is None:
        force_recategorize_websites = []
    
    client = authenticate_google_sheets()
    sheet = client.open("Website Categorization Test").sheet1
    llm = ChatOpenAI(model="gpt-4", temperature=0, api_key=OPENAI_API_KEY)
    rows = sheet.get_all_values()
    batch_updates = []  # Store updates for batch processing
    batch_size = 10  # Process 10 rows at a time

    print(f"Found {len(rows)} total rows in sheet")
    print(f"Force recategorizing these websites: {force_recategorize_websites}")
    print("First few rows of data:")
    for row in rows[:5]:
        print(row)

    for i, row in enumerate(rows[1:], start=2):
        if len(row) < 5:
            print(f"Skipping row {i} - insufficient columns: {row}")
            continue
        if not row[0].strip():
            print(f"Skipping row {i} - no website URL")
            continue
            
        website = row[0]
        
        # Skip websites that already have primary and secondary category data
        # unless they're in the force_recategorize_websites list
        if len(row) >= 3 and row[1].strip() and row[2].strip() and website not in force_recategorize_websites:
            print(f"Skipping row {i} - already has category data: Primary: {row[1]}, Secondary: {row[2]}")
            continue

        print(f"\n🔍 Processing website: {website}")

        try:
            # Fetch website content first
            print(f"Fetching content for {website}...")
            content = await fetch_website_content(website)
            
            # Categorize using LLM with the fetched content
            print(f"Attempting LLM categorization...")
            result = categorize_website(llm, content)
            print(f"LLM categorization result status: {result['status']}")
            
            primary_category, secondary_categories, niche, status = result["primary_category"], result["secondary_categories"], result["niche"], result["status"]

            # If LLM fails, use web crawling
            if status == "Failed":
                print(f"🔄 LLM Failed, Crawling {website}...")
                crawl_result = await simple_crawl(website)
                primary_category, secondary_categories, niche = crawl_result["primary_category"], crawl_result["secondary_categories"], crawl_result["niche"]

            # Add to batch updates
            batch_updates.append({
                'range': f'B{i}:D{i}',
                'values': [[primary_category, secondary_categories, niche]]
            })

            # Process batch when it reaches batch_size
            if len(batch_updates) >= batch_size:
                try:
                    sheet.batch_update(batch_updates)
                    print(f"✅ Batch updated {len(batch_updates)} rows")
                    batch_updates = []  # Clear the batch
                    time.sleep(61)  # Wait 61 seconds between batches to respect rate limits
                except gspread.exceptions.APIError as e:
                    if 'Quota exceeded' in str(e):
                        print("Rate limit hit, waiting 120 seconds...")
                        time.sleep(120)  # Wait longer if we hit the rate limit
                        # Retry the batch
                        sheet.batch_update(batch_updates)
                        batch_updates = []
                    else:
                        raise e

        except Exception as e:
            print(f"Error processing {website}: {str(e)}")
            continue

    # Process any remaining updates
    if batch_updates:
        try:
            sheet.batch_update(batch_updates)
            print(f"✅ Final batch updated {len(batch_updates)} rows")
        except gspread.exceptions.APIError as e:
            if 'Quota exceeded' in str(e):
                print("Rate limit hit on final batch, waiting 120 seconds...")
                time.sleep(120)
                sheet.batch_update(batch_updates)
            else:
                raise e

# Add token counting function
def estimate_tokens(text):
    """Estimate the number of tokens in a text string."""
    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
        return len(encoding.encode(text))
    except:
        # Fallback estimation if tiktoken is not available
        return len(text.split()) * 1.3  # Rough estimate: 1 word ≈ 1.3 tokens

async def process_single_website(website_url):
    """Process a single website for testing purposes.
    
    Args:
        website_url: The URL of the website to process
        
    Returns:
        dict: The categorization result
    """
    print(f"🔍 Processing website: {website_url}")
    
    try:
        # Initialize LLM
        llm = ChatOpenAI(model="gpt-4", temperature=0, api_key=OPENAI_API_KEY)
        
        # Fetch website content
        print(f"Fetching content for {website_url}...")
        content = await fetch_website_content(website_url)
        
        # Categorize using LLM
        print(f"Attempting LLM categorization...")
        result = categorize_website(llm, content)
        print(f"LLM categorization result status: {result['status']}")
        
        # If LLM fails, use web crawling
        if result["status"] == "Failed":
            print(f"🔄 LLM Failed, Crawling {website_url}...")
            crawl_result = await simple_crawl(website_url)
            result = crawl_result
        
        # Print the result
        print("\n✅ Categorization Result:")
        print(f"Primary Category: {result['primary_category']}")
        print(f"Secondary Categories: {result['secondary_categories']}")
        print(f"Niche: {result['niche']}")
        print(f"Status: {result['status']}")
        
        return result
        
    except Exception as e:
        print(f"Error processing {website_url}: {str(e)}")
        return {
            "primary_category": "Error", 
            "secondary_categories": "None", 
            "niche": "No", 
            "status": f"Error: {str(e)}"
        }

if __name__ == "__main__":
    # To recategorize specific websites, add them to the list
    # For example: asyncio.run(update_google_sheet(["opendoorhealth.com"]))
    # To process only websites without categories, use: asyncio.run(update_google_sheet())
    
    # Process all websites in the sheet that don't have categories
    asyncio.run(update_google_sheet())
    
    # Uncomment to process specific websites even if they already have categories
    # asyncio.run(update_google_sheet(["opendoorhealth.com"]))
    
    # Uncomment to test a single website without updating the sheet
    # asyncio.run(process_single_website("opendoorhealth.com"))