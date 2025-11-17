import os
os.environ.setdefault("LLAMA_LOG_LEVEL", "ERROR")
from llama_cpp import Llama
from app.retriever import HybridRetriever
import re

class RAGPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.llm = Llama(
            model_path=cfg["model"]["gguf_path"],
            n_ctx=cfg["model"]["ctx_tokens"],
            n_threads=cfg["model"]["n_threads"],
            n_gpu_layers=cfg["model"]["n_gpu_layers"],
            verbose=False,
        )
        self.retriever = HybridRetriever(cfg)

    def query(self, question: str, prompt_suffix: str = "", explain: bool = False, generate_followups: bool = True):
        retrieved = self.retriever.search(question,
                                          self.cfg["retrieval"]["top_k"],
                                          self.cfg["retrieval"]["rerank_k"],
                                          self.cfg["retrieval"]["hybrid_alpha"],
                                          explain=explain)
        
        # Extract sources for later display (keep full text for detailed output)
        sources = [{
            "id": i + 1,
            "title": r["title"],
            "score": r["score"],
            "text_full": r["text"],  # full chunk
            "text_snippet": r["text"][:200],  # snippet for preview
            "explain": r.get("explain", {})
        } for i, r in enumerate(retrieved)]
        
        context_text = "\n\n".join([f"[Source {i+1}] {r['title']}\n{r['text'][:400]}" for i, r in enumerate(retrieved)])
        # use system prompt from config if provided
        system_prompt = self.cfg.get("model", {}).get("system_prompt", "You are a helpful assistant.")
        
        # First generation: answer only (no follow-up suffix)
        prompt = (
            f"{system_prompt}\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {question}{prompt_suffix}\n\n"
            f"Instructions:\n"
            f"- Write naturally without explicitly mentioning sources (avoid phrases like 'According to Source 1' or 'Source 2 states').\n"
            f"- After each sentence or claim, add the citation tag [Source N] at the end.\n"
            f"- If you use information not present in the Context, tag it with [External Knowledge] at the end of that sentence.\n"
            f"- Use ONLY information from the Context above unless absolutely necessary.\n\n"
            f"Answer:\n"
        )

        prompt_tokens = len(prompt.split())
        ctx_limit = self.cfg["model"]["ctx_tokens"]
        # Reserve buffer for stability and allow larger generations (up to 2048 tokens)
        available = max(0, ctx_limit - prompt_tokens - 200)
        safe_max_output = min(2048, max(256, available))
        print(f"🧩 Prompt length: {prompt_tokens} tokens (limit {ctx_limit}), max output cap: {safe_max_output}")

        try:
            response = self.llm(
                prompt,
                max_tokens=safe_max_output,
                temperature=self.cfg["model"]["temperature"],
                top_p=self.cfg["model"]["top_p"],
                repeat_penalty=self.cfg["model"]["repeat_penalty"],
            )
            answer = response["choices"][0]["text"].strip()
            
            # Clean up answer: remove duplicate "Answer:" labels
            answer = re.sub(r'^Answer:\s*', '', answer, flags=re.IGNORECASE)
            answer = re.sub(r'\n\s*Answer:\s*', '\n', answer, flags=re.IGNORECASE)
            
            # Extract citations and build evidence map
            evidence_map = self._extract_evidence_map(answer, sources)
            
            # LLM-based relevance explanations (only when explain=True for --justify flag)
            if explain:
                for source in sources:
                    llm_reasoning = self._explain_relevance_llm(question, source.get('text_full', '')[:500])
                    # Add to existing explain dict
                    if 'explain' in source:
                        source['explain']['llm_reasoning'] = llm_reasoning
                    else:
                        source['explain'] = {'llm_reasoning': llm_reasoning}
            
            # Detect external knowledge usage
            has_external = "[External Knowledge]" in answer or "[External]" in answer
            
            # Detect uncited claims (sentences without [Source N] citations)
            uncited_warning = self._check_for_uncited_claims(answer)
            
            # Second generation: follow-up questions if requested
            followup_questions = ""
            if generate_followups:
                followup_questions = self._generate_followups(question, answer, context_text)
            
            return {
                "answer": answer,
                "sources": sources,
                "evidence_map": evidence_map,
                "has_external_knowledge": has_external,
                "uncited_warning": uncited_warning,
                "followup_questions": followup_questions
            }
        except Exception as e:
            return {"answer": f"❌ LLM error: {e}", "sources": [], "evidence_map": [], "has_external_knowledge": False, "uncited_warning": False, "followup_questions": ""}
            
            # Detect uncited claims (sentences without [Source N] citations)
            uncited_warning = self._check_for_uncited_claims(answer)
            
            return {
                "answer": answer,
                "sources": sources,
                "evidence_map": evidence_map,
                "has_external_knowledge": has_external,
                "uncited_warning": uncited_warning
            }
        except Exception as e:
            return {"answer": f"❌ LLM error: {e}", "sources": [], "evidence_map": [], "has_external_knowledge": False, "uncited_warning": False}
    
    def _extract_evidence_map(self, answer: str, sources: list) -> list:
        """Extract citations from answer and map them to source spans."""
        evidence = []
        
        # Find all citations like [Source 1], [Source 2], etc.
        citation_pattern = r'\[Source (\d+)\]'
        citations = re.findall(citation_pattern, answer)
        
        # Track which sources were cited
        cited_sources = set(int(c) for c in citations)
        
        for source_id in sorted(cited_sources):
            # Find corresponding source (source_id is 1-indexed)
            source = next((s for s in sources if s["id"] == source_id), None)
            if source:
                # Extract a representative span (first 200 chars of the chunk)
                span = source["text_full"][:200] + "..." if len(source["text_full"]) > 200 else source["text_full"]
                evidence.append({
                    "id": source_id,
                    "source_title": source["title"],
                    "span": span
                })
        
        return evidence
    
    def _check_for_uncited_claims(self, answer: str) -> bool:
        """Check if answer contains claims without source citations.
        Treat multi-line absence disclaimers as safe and avoid false positives on formatting fragments."""
        # Normalize newlines and collapse placeholder spacing
        normalized = re.sub(r'[\r]+', '', answer)
        # Split on sentence boundaries preserving disclaimers over multiple lines
        raw_sentences = re.split(r'(?<=[.!?])\s+', normalized)
        uncited_count = 0

        for sent in raw_sentences:
            s = sent.strip()
            if not s:
                continue
            # Skip follow-up section entirely
            if s.lower().startswith("follow-up questions:"):
                continue
            if re.match(r'^[123]\.\s', s):
                continue
            # Merge lines that were broken mid-sentence (heuristic: remove internal newlines)
            s = s.replace('\n', ' ').strip()
            # Disclaimers considered safe
            disclaimer_phrases = [
                "do not mention",
                "not mentioned",
                "sources do not",
                "provided sources",
                "the provided sources do not",
            ]
            is_disclaimer = any(p in s.lower() for p in disclaimer_phrases)
            has_citation = bool(re.search(r'\[Source \d+\]', s))
            has_external_marker = "[External" in s

            # Count only substantive sentences likely to assert facts without grounding
            if not has_citation and not has_external_marker and not is_disclaimer and len(s.split()) > 6:
                uncited_count += 1

        return uncited_count > 0
    
    def _generate_followups(self, question: str, answer: str, context: str) -> str:
        """Generate follow-up questions in a separate LLM call based on the answer and context."""
        followup_prompt = f"""Based on this question and answer, suggest up to 3 useful follow-up questions that can be answered using the provided context sources.

Original Question: {question}

Answer: {answer}

Context Sources:
{context[:800]}

Provide 3 concise follow-up questions as a numbered list (1., 2., 3.). Only suggest questions that are answerable from the context."""
        
        try:
            response = self.llm(
                followup_prompt,
                max_tokens=150,
                temperature=0.7,
                top_p=0.95,
                repeat_penalty=1.1,
            )
            followups = response["choices"][0]["text"].strip()
            # Clean up any leading "Follow-up questions:" label
            followups = re.sub(r'^Follow-up questions?:\s*', '', followups, flags=re.IGNORECASE)
            return followups
        except Exception:
            return ""
    
    def _explain_relevance_llm(self, query: str, chunk: str) -> str:
        """Use LLM to generate a natural language explanation of why this chunk is relevant."""
        prompt = f"""Query: "{query}"
Source excerpt: "{chunk}"

In one concise sentence, explain why this source is relevant to the query. Focus on what specific information it provides.
Explanation:"""
        
        try:
            response = self.llm(
                prompt,
                max_tokens=60,
                temperature=0.3,  # Lower temp for focused explanations
                top_p=0.9,
                repeat_penalty=1.1,
            )
            explanation = response["choices"][0]["text"].strip()
            # Clean up any repeated prompt text
            explanation = re.sub(r'^(Explanation:|Why relevant:)\s*', '', explanation, flags=re.IGNORECASE)
            return explanation if explanation else "Provides relevant contextual information."
        except Exception as e:
            return f"Error generating explanation: {str(e)[:50]}"
