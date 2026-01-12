# examples/langchain_quickstart.py

import sys
import os
sys.path.append(os.getcwd())

import time

try:
    from langchain.llms.fake import FakeListLLM
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    from orchestra import enhance
    HAS_LANGCHAIN = True
except ImportError as e:
    HAS_LANGCHAIN = False
    MISSING_DEP_ERROR = e

if __name__ == "__main__":
    if not HAS_LANGCHAIN:
        print(f"⚠️  LangChain not installed. Skipping example.")
        print(f"Error: {MISSING_DEP_ERROR}")
        print("Run: pip install langchain")
        sys.exit(0)

    # 1. Create a Fake LLM (for demo speed) but treat usage as "expensive"
    llm = FakeListLLM(responses=["Berlin", "Paris", "Tokyo"])

    # 2. Create Chain
    prompt = PromptTemplate(template="What is the capital of {country}?", input_variables=["country"])
    chain = LLMChain(llm=llm, prompt=prompt)

    # 3. Enhance
    cached_chain = enhance(chain) # ✨

    # 4. Run
    print("\n--- Run 1: Cold ---")
    start = time.time()
    # Simulate "work" by sleeping manually for this demo since FakeListLLM is fast
    time.sleep(1.5) 
    print(cached_chain.run("Germany"))
    print(f"Time: {time.time() - start:.2f}s")

    print("\n--- Run 2: Warm (Cached) ---")
    start = time.time()
    # No sleep here, cache intercepts before execution!
    print(cached_chain.run("Germany"))
    print(f"Time: {time.time() - start:.2f}s")
