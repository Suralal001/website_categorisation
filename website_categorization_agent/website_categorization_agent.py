import asyncio
from langchain.chat_models import ChatOpenAI

async def update_google_sheet():
    """Updates missing categories in Google Sheets."""
    client = authenticate_google_sheets()
    sheet = client.open("Website Categorization Test").sheet1
    llm = ChatOpenAI(model="gpt-4", temperature=0, api_key=OPENAI_API_KEY)
    rows = sheet.get_all_values()

    print(f"Found {len(rows)} total rows in sheet")
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

        website, primary_category, secondary_categories, niche, description = row[0], row[1], row[2], row[3], row[4]

        # Skip if already categorized
        if primary_category and secondary_categories and niche and description:
            print(f"Skipping row {i} - already categorized: {website}")
            continue

        print(f"🔍 Processing: {website}")

        try:
            # Categorize using LLM
            result = categorize_website(llm, website)
            primary_category, secondary_categories, niche, status = result["primary_category"], result["secondary_categories"], result["niche"], result["status"]

            # If LLM fails, use web crawling
            if status == "Failed":
                print(f"🔄 LLM Failed, Crawling {website}...")
                crawl_result = await simple_crawl(website)
                primary_category, secondary_categories, niche = crawl_result["primary_category"], crawl_result["secondary_categories"], crawl_result["niche"]

            # Update the sheet directly
            sheet.update(f'B{i}:D{i}', [[primary_category, secondary_categories, niche]])
            print(f"✅ Updated row {i}: {website}")

        except Exception as e:
            print(f"Error processing {website}: {str(e)}")
            continue

if __name__ == "__main__":
    asyncio.run(update_google_sheet()) 