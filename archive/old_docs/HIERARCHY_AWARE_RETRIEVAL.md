# Hierarchy-Aware Retrieval Strategy

## Overview
The RAPTOR hierarchy enables intelligent retrieval based on question type. This document explains how different question types should use different hierarchy levels.

## Retrieval Strategies by Question Type

### 1. **Factual Questions** → Level 0 (Base Chunks)
**Examples:**
- "What is the learning rate used in the experiment?"
- "Who are the authors of this paper?"
- "What dataset was used for evaluation?"

**Strategy:**
```python
# Retrieve from Level 0 only
chunks = retrieve_from_level(question, level=0, top_k=5)
```

**Reasoning:**
- Specific facts are only in raw chunks
- Higher levels abstract away details
- Need exact quotes and specific values

---

### 2. **Summary Questions** → Level 2/3 (High-Level Summaries)
**Examples:**
- "Summarize all documents"
- "What are the main points?"
- "Give me an overview"

**Strategy:**
```python
# For single-doc summary: retrieve from Level 1-2 for that doc
# For all-docs summary: retrieve from Level 2-3 (global summaries)
chunks = retrieve_from_level(question, level=2, top_k=3)
```

**Reasoning:**
- Pre-computed abstractions already exist
- No need to re-summarize from Level 0
- Higher levels = broader context

**Special Case: "Summarize All Documents"**
```python
def summarize_all_documents(self):
    # Retrieve ALL Level 2 summaries (no vector search, just get all)
    conn = psycopg2.connect(**self.db_config)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT content, metadata->>'source_file' as source
            FROM documents
            WHERE user_id = %s AND hierarchy_level = 2
            ORDER BY metadata->>'source_file'
        """, (self.user_id,))
        summaries = cur.fetchall()
    
    # Combine and send to Gemini for final synthesis
    combined = "\n\n".join([f"[{s['source']}]\n{s['content']}" for s in summaries])
    prompt = f"Synthesize these document summaries:\n\n{combined}\n\nProvide a comprehensive overview:"
    return self.gemini_model.generate_content(prompt).text
```

---

### 3. **Comparison Questions** → Multi-Level (L1 + L0)
**Examples:**
- "Compare method A and method B"
- "What are the differences between these approaches?"
- "How do the results differ?"

**Strategy:**
```python
# Step 1: Find relevant clusters at Level 1
level1_chunks = retrieve_from_level(question, level=1, top_k=5)

# Step 2: Expand to Level 0 (get children of relevant clusters)
level0_chunks = []
for l1_chunk in level1_chunks:
    children_ids = l1_chunk['metadata'].get('children', [])
    level0_chunks.extend(retrieve_by_ids(children_ids))

# Step 3: Use both levels for context
context = level1_chunks + level0_chunks[:10]  # Mix high-level + details
```

**Reasoning:**
- Level 1 summaries help identify relevant clusters
- Level 0 provides specific evidence for comparison
- Hybrid approach balances breadth and depth

---

### 4. **Analytical Questions** → Adaptive Multi-Level
**Examples:**
- "Why does this approach work better?"
- "How does the transformer architecture improve performance?"
- "What are the implications of this finding?"

**Strategy:**
```python
# Adaptive: retrieve from multiple levels
chunks_l0 = retrieve_from_level(question, level=0, top_k=5)  # Details
chunks_l1 = retrieve_from_level(question, level=1, top_k=3)  # Context
chunks_l2 = retrieve_from_level(question, level=2, top_k=2)  # Big picture

# Combine and deduplicate
context = chunks_l2 + chunks_l1 + chunks_l0
```

**Reasoning:**
- Need both high-level patterns and specific evidence
- Level 2 provides "why" (big picture)
- Level 0 provides "how" (specific mechanisms)

---

## Implementation in raptor_qa.py

### Current Issue
The system has complex retrieval logic but doesn't fully leverage the hierarchy. It sometimes bypasses hierarchy and retrieves only from Level 0.

### Proposed Fix

```python
def ask(self, question: str) -> str:
    """Main entry point with hierarchy-aware retrieval"""
    
    # 1. Classify question type
    q_type = self.classify_question(question)
    print(f"[Question Type: {q_type.value}]")
    
    # 2. Retrieve based on question type
    if q_type == QuestionType.FACTUAL:
        context_chunks = self.retrieve_from_level(question, level=0, top_k=5)
    
    elif q_type == QuestionType.SUMMARY:
        # Check if asking for "all documents"
        if re.search(r'\b(all|every|each)\b', question.lower()):
            return self.summarize_all_documents()
        else:
            context_chunks = self.retrieve_from_level(question, level=2, top_k=3)
    
    elif q_type == QuestionType.COMPARISON:
        context_chunks = self.retrieve_comparison(question)
    
    elif q_type == QuestionType.ANALYTICAL:
        context_chunks = self.retrieve_multi_level(question, levels=[0, 1, 2], k_per_level=3)
    
    # 3. Generate answer with context
    return self.generate_answer(question, context_chunks)


def summarize_all_documents(self) -> str:
    """Use Level 2/3 summaries instead of re-summarizing Level 0"""
    conn = psycopg2.connect(**self.db_config)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Get all Level 2 summaries (high-level)
            cur.execute("""
                SELECT content, metadata
                FROM documents
                WHERE user_id = %s AND hierarchy_level = 2
                ORDER BY metadata->>'source_file'
            """, (self.user_id,))
            summaries = cur.fetchall()
            
            if not summaries:
                # Fallback: use Level 1 if no Level 2
                cur.execute("""
                    SELECT content, metadata
                    FROM documents
                    WHERE user_id = %s AND hierarchy_level = 1
                    ORDER BY metadata->>'source_file'
                """, (self.user_id,))
                summaries = cur.fetchall()
            
            # Combine summaries
            combined = "\n\n---\n\n".join([
                f"[Source: {s['metadata'].get('source_file', 'unknown')}]\n{s['content']}"
                for s in summaries
            ])
            
            prompt = f"""You are a research assistant. Synthesize these document summaries into a comprehensive overview:

{combined}

Provide a clear, well-organized summary covering:
1. Main topics/themes
2. Key findings
3. Important methodologies or approaches

Summary:"""
            
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip()
    finally:
        conn.close()


def retrieve_comparison(self, question: str) -> List[Dict]:
    """Multi-level retrieval for comparison questions"""
    # Step 1: Find relevant Level 1 clusters
    level1_chunks = self.retrieve_from_level(question, level=1, top_k=5)
    
    # Step 2: Expand to Level 0 (get children)
    level0_chunks = []
    conn = psycopg2.connect(**self.db_config)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            for l1_chunk in level1_chunks:
                children_ids = l1_chunk['metadata'].get('children', [])
                if children_ids:
                    cur.execute("""
                        SELECT doc_id, content, metadata, hierarchy_level
                        FROM documents
                        WHERE user_id = %s AND doc_id = ANY(%s)
                        LIMIT 10
                    """, (self.user_id, children_ids))
                    level0_chunks.extend(cur.fetchall())
    finally:
        conn.close()
    
    # Combine: Level 1 summaries + Level 0 details
    return level1_chunks + level0_chunks[:10]
```

## Benefits of Hierarchy-Aware Retrieval

1. **Efficiency:** Don't retrieve 100+ Level 0 chunks for summary questions
2. **Quality:** Level 2 summaries are already coherent, no re-summarization needed
3. **Consistency:** Same hierarchy structure for all question types
4. **Scalability:** Works for 5 documents or 500 documents

## Testing Strategy

```python
# Test each question type
qa = RaptorQASystem(user_id="...")

# Factual → Level 0
print(qa.ask("What learning rate was used?"))

# Summary → Level 2/3
print(qa.ask("Summarize all documents"))

# Comparison → Multi-level
print(qa.ask("Compare the two methods described"))

# Analytical → Adaptive
print(qa.ask("Why does this approach work?"))
```

---

**Next Steps:**
1. Update `raptor_qa.py` with hierarchy-aware retrieval
2. Test each question type
3. Remove old simplified approaches (raptor_qa_simplified.py)
4. Document incremental update strategies
