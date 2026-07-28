import os
import json
import asyncio
from pathlib import Path
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()
from config import config

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
TOPICS_FILE = DATA_DIR / "topics_list.json"

async def get_topics(llm):
    print("Generating 100 popular debate topics...")
    prompt = (
        "Generate a JSON array of exactly 100 popular, diverse, and thought-provoking debate topics. "
        "Each topic should be a question (e.g., 'Is universal basic income a good idea?'). "
        "Output ONLY valid JSON, starting with '[' and ending with ']'. No other text."
    )
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    content = response.content.strip()
    
    # Clean up if the model wrapped it in markdown code blocks
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()
    
    try:
        topics = json.loads(content)
        if len(topics) > 100:
            topics = topics[:100]
        print(f"Successfully generated {len(topics)} topics.")
        return topics
    except Exception as e:
        print(f"Failed to parse JSON. Error: {e}")
        print(f"Raw output: {content[:200]}...")
        # Fallback to a small list for testing if parsing fails
        return [
            "Is artificial intelligence a threat to humanity?",
            "Should higher education be free for everyone?",
            "Is space exploration worth the cost?",
            "Should a four-day work week be standard?"
        ]

async def generate_facts_for_topic(llm, topic, sem, idx, total):
    async with sem:
        print(f"[{idx}/{total}] Generating facts for: {topic}")
        prompt = (
            f"Write a short, highly factual, 2-paragraph research summary for the debate topic: '{topic}'\n"
            "Include concrete statistics, numbers, and key arguments for both sides. "
            "Keep it dense and informative. Do not use conversational filler."
        )
        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            
            # Save to a file named after the topic (slugified)
            slug = "".join(c if c.isalnum() else "_" for c in topic.lower())
            # Truncate slug if too long
            slug = slug[:50].strip("_")
            
            file_path = DATA_DIR / f"{slug}.txt"
            file_path.write_text(f"Topic: {topic}\n\n{response.content}", encoding="utf-8")
        except Exception as e:
            print(f"Error generating facts for '{topic}': {e}")
        
        # Slight delay to avoid hitting rate limits too hard
        await asyncio.sleep(0.5)

async def main():
    if not config.groq_api_key:
        print("No GROQ_API_KEY found. Cannot generate topics.")
        return

    # Use a faster/smaller model for generation if desired, but versatile is fine
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7, api_key=config.groq_api_key)
    
    # 1. Get Topics
    topics = await get_topics(llm)
    
    # Save topics list
    with open(TOPICS_FILE, "w", encoding="utf-8") as f:
        json.dump(topics, f, indent=2)
        
    # 2. Generate Data Files
    # Limit concurrency to 3 to avoid aggressive rate limits (Groq allows a fair bit, but 3 is safe)
    sem = asyncio.Semaphore(3)
    tasks = []
    
    for i, topic in enumerate(topics, start=1):
        tasks.append(generate_facts_for_topic(llm, topic, sem, i, len(topics)))
        
    await asyncio.gather(*tasks)
    
    print("Finished populating the database!")

if __name__ == "__main__":
    asyncio.run(main())
