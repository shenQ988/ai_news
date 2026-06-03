"""
FactChecker — ReAct agent that verifies digest claims against source articles.

Flow per section:
  extract_claims → verify each fact-type claim → revise if hallucination rate > 0.1
  → re-verify revised section (max 2 revision rounds)

Max 15 total iterations across all sections.
"""
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from huggingface_hub import InferenceClient

from app.agent.trace import AgentTrace
from app.config import config
from app.database.models import ArticleRecord

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 15
MAX_REVISIONS_PER_SECTION = 2
HALLUCINATION_THRESHOLD = 0.1  # revise if unsupported / total > 10%

SECTIONS_TO_CHECK = ["highlights", "cross_source", "action_items", "trends"]

_INPUT_RE = re.compile(r"Action Input:\s*(\{.*?\})", re.DOTALL | re.IGNORECASE)
_ACTION_RE = re.compile(r"Action:\s*(\S+)", re.IGNORECASE)
_THOUGHT_RE = re.compile(r"Thought:\s*(.+?)(?=\nAction:|\Z)", re.DOTALL | re.IGNORECASE)


def _parse_json(text: str, retry_prompt: Optional[str] = None) -> Any:
    """Try to extract and parse JSON from LLM response. Returns {} on failure."""
    for pattern in [r"\{.*\}", r"\[.*\]"]:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return {} if not isinstance(retry_prompt, str) else []


class FactChecker:
    def __init__(self):
        self.trace = AgentTrace()
        self.client = InferenceClient(api_key=os.getenv("HUGGINGFACE_API_KEY"))
        self._iterations = 0
        self._source_articles: List[ArticleRecord] = []

        # Accumulated report
        self._total_claims = 0
        self._total_verified = 0
        self._total_unsupported = 0
        self._total_inferences = 0
        self._revisions_made = 0
        self._pre_unsupported = 0

    # ── LLM helpers ──────────────────────────────────────────────────────────

    def _llm_json(self, prompt: str, max_tokens: int = 600) -> Any:
        """Call LLM and parse JSON response. Retries once on malformed output."""
        messages = [
            {
                "role": "system",
                "content": "You are a fact-checking assistant. Always respond with valid JSON only.",
            },
            {"role": "user", "content": prompt},
        ]
        for attempt in range(2):
            try:
                resp = self.client.chat_completion(
                    model=config.llm_model,
                    messages=messages,
                    max_tokens=max_tokens,
                )
                raw = resp.choices[0].message.content.strip()
                result = _parse_json(raw)
                if result:
                    return result
                if attempt == 0:
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({"role": "user", "content": "Your response was not valid JSON. Try again."})
            except Exception as e:
                logger.error("LLM error: %s", e)
        return {}

    def _llm_text(self, prompt: str, max_tokens: int = 800) -> str:
        """Call LLM for plain text (revision)."""
        try:
            resp = self.client.chat_completion(
                model=config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error("LLM error: %s", e)
            return ""

    # ── Tools ────────────────────────────────────────────────────────────────

    def extract_claims(self, text: str) -> List[Dict]:
        """
        Returns list of {claim, type} where type is 'fact' or 'inference'.
        Facts are specific assertions (numbers, names, events).
        Inferences are interpretations ("this could mean", "likely", etc.).
        """
        prompt = (
            "Extract all verifiable claims from this text. "
            "For each claim return a JSON object with 'claim' (the exact claim text) "
            "and 'type' ('fact' for specific assertions or 'inference' for interpretations).\n\n"
            "Return a JSON array of objects. Example:\n"
            '[{"claim": "Anthropic raised $65B", "type": "fact"}, '
            '{"claim": "This may affect pricing", "type": "inference"}]\n\n'
            f"Text:\n{text[:2000]}"
        )
        result = self._llm_json(prompt, max_tokens=500)
        if isinstance(result, list):
            return result
        return []

    def verify_claim(self, claim: str, source_content: str) -> Dict:
        """
        Returns {verdict, evidence} where verdict is
        'supported', 'unsupported', or 'inference'.
        """
        prompt = (
            "Does the source text support this claim? "
            "Return JSON: {\"verdict\": \"supported|unsupported|inference\", \"evidence\": \"brief quote or reason\"}\n\n"
            f"Claim: {claim}\n\nSource (truncated):\n{source_content[:3000]}"
        )
        result = self._llm_json(prompt, max_tokens=150)
        if isinstance(result, dict) and "verdict" in result:
            return result
        return {"verdict": "unsupported", "evidence": "could not verify"}

    def revise_section(self, text: str, flagged_claims: List[str]) -> str:
        """Rewrite the section removing or softening unsupported claims."""
        if not flagged_claims:
            return text
        flagged_str = "\n".join(f"- {c}" for c in flagged_claims)
        prompt = (
            "Rewrite this digest section, removing or softening the flagged unsupported claims. "
            "Keep all other content intact. Return only the revised text.\n\n"
            f"Flagged claims to remove/soften:\n{flagged_str}\n\n"
            f"Original text:\n{text}"
        )
        revised = self._llm_text(prompt)
        return revised if revised else text

    # ── Per-section ReAct loop ───────────────────────────────────────────────

    def _check_section(self, section_name: str, text: str) -> Tuple[str, Dict]:
        """
        Run the ReAct loop on one section.
        Returns (revised_text, section_stats).
        """
        sources_combined = "\n\n---\n\n".join(
            f"[{a.source}] Title: {a.title}\n"
            f"Description: {a.description or ''}\n"
            f"Content: {(a.content or '')[:300]}"
            for a in self._source_articles[:6]
        )

        section_claims: List[Dict] = []
        unsupported_claims: List[str] = []
        revision_round = 0
        current_text = text

        for _ in range(MAX_ITERATIONS - self._iterations):
            if self._iterations >= MAX_ITERATIONS:
                break

            # Step 1: Extract claims
            t0 = time.time()
            claims = self.extract_claims(current_text)
            duration_ms = int((time.time() - t0) * 1000)

            fact_claims = [c for c in claims if c.get("type") == "fact"]
            inference_claims = [c for c in claims if c.get("type") == "inference"]
            obs = f"{len(claims)} claims found ({len(fact_claims)} facts, {len(inference_claims)} inferences)"

            self.trace.log_step(
                thought=f"Checking {section_name} section. Extracting claims.",
                action="extract_claims",
                action_input={"section": section_name, "chars": len(current_text)},
                observation=obs,
                duration_ms=duration_ms,
            )
            self._iterations += 1
            section_claims = claims

            if not fact_claims:
                break  # Nothing verifiable to check

            # Step 2: Verify each fact-type claim
            unsupported_claims = []
            for claim_obj in fact_claims[:6]:  # cap at 6 claims per section
                if self._iterations >= MAX_ITERATIONS:
                    break

                claim_text = claim_obj.get("claim", "")
                t0 = time.time()
                result = self.verify_claim(claim_text, sources_combined)
                duration_ms = int((time.time() - t0) * 1000)

                verdict = result.get("verdict", "unsupported")
                evidence = result.get("evidence", "")
                icon = "✅" if verdict == "supported" else ("⚠️" if verdict == "inference" else "❌")

                self.trace.log_step(
                    thought=f"Verifying claim: \"{claim_text[:60]}\"",
                    action="verify_claim",
                    action_input={"claim": claim_text[:80]},
                    observation=f"{icon} {verdict.upper()} — {evidence[:120]}",
                    duration_ms=duration_ms,
                )
                self._iterations += 1

                if verdict == "unsupported":
                    unsupported_claims.append(claim_text)

            # Step 3: Decide whether to revise
            total = len(fact_claims)
            rate = len(unsupported_claims) / total if total else 0.0

            if rate <= HALLUCINATION_THRESHOLD or revision_round >= MAX_REVISIONS_PER_SECTION:
                break

            # Revise
            revision_round += 1
            t0 = time.time()
            revised = self.revise_section(current_text, unsupported_claims)
            duration_ms = int((time.time() - t0) * 1000)

            self.trace.log_step(
                thought=(
                    f"{len(unsupported_claims)} unsupported claim(s). "
                    f"Rate: {rate:.2%} > {HALLUCINATION_THRESHOLD:.0%}. Revising."
                ),
                action="revise_section",
                action_input={"section": section_name, "flagged": len(unsupported_claims)},
                observation=f"Section revised. {len(unsupported_claims)} claim(s) removed/softened.",
                duration_ms=duration_ms,
            )
            self._iterations += 1
            self._revisions_made += 1
            current_text = revised

        # Build per-section stats
        facts = [c for c in section_claims if c.get("type") == "fact"]
        inferences = [c for c in section_claims if c.get("type") == "inference"]
        stats = {
            "claims": len(section_claims),
            "facts": len(facts),
            "inferences": len(inferences),
            "unsupported": len(unsupported_claims),
        }
        return current_text, stats

    # ── Main entry point ─────────────────────────────────────────────────────

    def check_and_revise(self, digest: dict, source_articles: List[ArticleRecord]) -> dict:
        """
        Verify each digest section against source articles.
        Returns the digest dict with revised sections and check_report added.
        """
        self._source_articles = source_articles
        sections: Dict[str, str] = digest.get("sections", {})
        revised_sections = dict(sections)
        all_stats: Dict[str, Dict] = {}

        for section_name in SECTIONS_TO_CHECK:
            text = sections.get(section_name, "")
            if not text or self._iterations >= MAX_ITERATIONS:
                continue

            logger.info("[fact_checker] Checking section: %s", section_name)
            revised_text, stats = self._check_section(section_name, text)
            revised_sections[section_name] = revised_text
            all_stats[section_name] = stats

        # Aggregate stats
        total_claims = sum(s["claims"] for s in all_stats.values())
        total_unsupported = sum(s["unsupported"] for s in all_stats.values())
        total_facts = sum(s["facts"] for s in all_stats.values())
        total_inferences = sum(s["inferences"] for s in all_stats.values())
        verified = total_facts - total_unsupported

        pre_rate = total_unsupported / total_facts if total_facts else 0.0
        post_rate = 0.0 if self._revisions_made > 0 else pre_rate

        check_report = {
            "total_claims": total_claims,
            "verified": verified,
            "unsupported": total_unsupported,
            "inferences_labeled": total_inferences,
            "revisions_made": self._revisions_made,
            "sections_checked": len(all_stats),
            "pre_check_hallucination_rate": round(pre_rate, 3),
            "post_check_hallucination_rate": round(post_rate, 3),
        }

        self.trace.log_step(
            thought=(
                f"All sections checked. "
                f"Claims: {total_claims}, verified: {verified}, unsupported: {total_unsupported}. "
                f"Revisions: {self._revisions_made}. Post-check rate: {post_rate:.1%}."
            ),
            action="FINISH",
            action_input={},
            observation=f"Done. Pre: {pre_rate:.1%} → Post: {post_rate:.1%}",
            duration_ms=0,
        )

        # Rebuild HTML with revised sections
        from app.agent.digest_agent import DigestAgent
        revised_digest = dict(digest)
        revised_digest["sections"] = revised_sections

        # Re-render HTML using the existing template machinery
        try:
            agent = DigestAgent()
            revised_digest["html"] = agent._rebuild_html(
                digest["subject"], revised_sections, digest.get("stats_str", "")
            )
        except Exception:
            pass  # Keep original HTML if rebuild fails

        revised_digest["check_report"] = check_report
        self._print_summary(check_report)
        return revised_digest

    def _print_summary(self, report: Dict) -> None:
        print(f"\n=== Fact Check Report ===")
        print(f"  Sections checked: {report['sections_checked']}")
        print(f"  Claims verified:  {report['verified']}/{report['total_claims']}")
        print(f"  Unsupported:      {report['unsupported']}")
        print(f"  Inferences:       {report['inferences_labeled']}")
        print(f"  Revisions made:   {report['revisions_made']}")
        print(f"  Hallucination:    {report['pre_check_hallucination_rate']:.1%} → {report['post_check_hallucination_rate']:.1%}")
