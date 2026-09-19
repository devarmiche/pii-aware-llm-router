# Business case — a sovereign, cost-aware document assistant

> This document is written *after* the measurements it cites, not before. Every number below is either read directly from `data/tracker_log.csv` / `eval/question_results.csv` / `eval/graded_results.csv`, or explicitly marked as an assumption with the reasoning behind it. Where a number can't be measured from this project's own runs, it is not stated as fact.

## 1. Business context

A European institution handling regulatory, legal or financial documents — e.g. a national supervisory authority, a public administration processing citizen files, or a company's legal/compliance function reviewing contracts and litigation — routinely asks an LLM to read a document and answer a question about it: "what is the fine amount", "summarize this clause", "does this contract expose us to X". Two constraints collide:

- **Data sovereignty.** The document may contain personal data (names, financial details, case parties) or business-sensitive information. Sending it to a US-operated API is a genuine legal and reputational exposure (see §2).
- **Cost and quality.** A fully sovereign, fully local setup (open-weight model on owned/EU-hosted infrastructure) is free per call but weaker on long-context, multi-document, or subtle legal reasoning — the kind of question this audience actually asks most often.

**pii-aware-llm-router** is a small reference implementation of the compromise most enterprise AI deployments actually reach: anonymize before anything leaves the local environment, then route based on an explicit, auditable rule — not a black-box classifier — between a local open-weight model and a frontier API model, with every call's tokens, latency and cost logged so the trade-off is a measured fact instead of a slide.

## 2. Risk and sovereignty analysis

**Legal basis.** Under the GDPR, transferring personal data to a country without an adequacy decision requires a valid transfer mechanism (Standard Contractual Clauses, etc.) *and*, per the CJEU's **Schrems II** ruling (Case C-311/18, 16 July 2020), a case-by-case assessment that the destination country's legal system — including its public authorities' access to data — offers a level of protection "essentially equivalent" to the EU's. The US **CLOUD Act** (2018) compels US-headquartered providers to produce data they control on US legal request, regardless of where that data is physically stored — which is precisely the kind of third-country public-authority access Schrems II says must be weighed, and which no amount of EU data-center hosting by a US provider fully neutralizes on its own.

**The CNIL's position** has consistently been that anonymization or pseudonymization performed *before* a transfer, such that the recipient cannot re-identify the data subject without information held only by the sender, takes the transfer outside GDPR Chapter V's international-transfer rules for the anonymized fields. This is the legal rationale for this project's architecture: masking PII locally, before any text reaches an external API, is not just a privacy nicety — it is what allows the "complex question, no PII" case to legitimately use a non-EU API at all.

**SecNumCloud** (ANSSI's qualification scheme) and equivalent sovereign-cloud labels exist precisely to give the "guaranteed no CLOUD-Act exposure" alternative: infrastructure operated by an entity immune to non-EU extraterritorial law. Positioning a French/EU model provider (Mistral AI) on sovereign infrastructure (OVHcloud, Scaleway, both SecNumCloud-qualified) as the default is the practical response to this constraint for anything that can't be masked or can't tolerate residual re-identification risk.

**What this project does and does not claim:**
- It does **not** claim GDPR compliance as a finished, auditable state — it is a proof of concept, not a DPIA.
- It **does** apply the principle the CNIL endorses (pre-transfer anonymization narrows Chapter V exposure) and makes the anonymizer's real, measured failure modes visible (§3) rather than assuming it is perfect.
- The router's "any detected PII forces local" rule (`src/router.py`) is a deliberately conservative sovereignty gate: even a question that looks purely factual is kept local the moment personal data is found in context, on the assumption that a false negative in the anonymizer is more costly than an unnecessary local call.

## 3. Anonymizer performance

Measured by `eval/eval_anonymizer.py` against `data/synthetic/gold_set.jsonl`: **30 synthetic French documents (HR letters, invoices, insurance claims, contract excerpts), 330 planted ground-truth entities**, generated together with their annotations so recall is measured against a known-complete answer, not a post-hoc labeling. Matching is overlap-based (a detection counts if its span intersects the gold span and the type matches), which is more forgiving than exact-boundary matching — see the script's docstring for why.

| Entity type | Gold | Predicted | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| DATE_TIME | 60 | 60 | 1.00 | 1.00 | 1.00 |
| EMAIL_ADDRESS | 30 | 30 | 1.00 | 1.00 | 1.00 |
| FR_NIR (social security number) | 30 | 30 | 1.00 | 1.00 | 1.00 |
| FR_SIRET | 30 | 30 | 1.00 | 1.00 | 1.00 |
| IBAN_CODE | 30 | 30 | 1.00 | 1.00 | 1.00 |
| PHONE_NUMBER | 30 | 31 | 0.97 | 1.00 | 0.98 |
| SALARY | 30 | 30 | 1.00 | 1.00 | 1.00 |
| PERSON | 60 | 107 | 0.56 | 1.00 | 0.72 |
| LOCATION | 30 | 92 | 0.37 | 0.80 | 0.51 |

**Read this as: recall is excellent everywhere (nothing sensitive that was planted got missed), but PERSON and LOCATION over-detect.** 107 PERSON predictions against 60 gold entities, and 92 LOCATION predictions against 30 gold entities, means the anonymizer is aggressive — it masks things that aren't PII (job titles used as if they were names, common nouns that are also place names) rather than missing real ones. For a masking system that feeds a downstream LLM, that is the safer failure direction (over-redaction degrades answer quality; under-redaction leaks data), but it is a real, measurable limitation: **35 spurious detections** (types like ORGANIZATION or URL, outside this project's PII taxonomy) were also produced and are not counted against precision/recall above since they were never meant to be masked in the first place — they are additional noise in the anonymized text the downstream model sees.

**PERSON recall by planted name origin** (the gold set deliberately includes hard cases — surnames that are common nouns/cities, hyphenated and particle names, and non-French-origin names — to test for a documented, common failure mode of NER systems: under-recognizing names outside the training distribution):

| Name category | Recall |
|---|---:|
| French (baseline) | 12/12 (1.00) |
| Common-noun/city surnames (Rose, Fleury, Lyon, Lemaire...) | 14/14 (1.00) |
| Hyphenated (Dupont-Moretti...) | 7/7 (1.00) |
| Particle names (de la Rochefoucauld...) | 4/4 (1.00) |
| Non-French-origin (Nguyen, Diallo, Kowalski, Ben Salah...) | 23/23 (1.00) |

On this gold set, `fr_core_news_md`'s NER does **not** show the recall drop by name origin that is commonly reported for this class of model — every category planted here was caught. This should be read as a property of this specific, small gold set (60 PERSON entities total) rather than a general claim; a larger and more adversarial name set could still surface a gap that 60 entities isn't powered to detect.

**Org names (`Société Générale` etc.):** not annotated as PII in the gold set and not detected as a dedicated type by the current recognizer set — a deliberate scope decision (see `claude.md`): organization names are treated as business context to keep, not personal data to mask, unless they double as an individual's employer in a context that makes the combination identifying (out of scope for this POC).

## 4. Router behavior

Every one of the 7 real corpus documents (CNIL sanctions, Conseil d'État decision, Schrems II judgment, EDPB guidelines, VINCI half-year report) contains detected PII — case parties, signatories, officials' names — so **the router's sovereignty gate fired on all 18 of 18 questions**, sending every single one to the local route in automatic mode, regardless of its a priori complexity tag (`factual_lookup` through `legal_interpretation`). On this corpus, the router's second-tier heuristic (document length, cross-referencing, legal keywords) never gets a chance to run: the PII gate dominates completely.

This is itself the headline finding of this project, not a footnote: **a real deployment on real regulatory/legal documents will overwhelmingly route to the free local model, not the paid API** — which is exactly the sovereignty-first design intent. But it only holds up if the local model can actually complete the request (§5).

To measure cost/latency/quality on both routes despite this — not just whatever the router happened to pick — `eval/run_questions.py` forces an answer from *each* route per question rather than trusting the auto-decision.

## 5. Cost model

**Real measured numbers**, from `eval/question_results.csv` (18 questions, forced through both routes):

| Route | n | Avg latency | Latency range | Avg cost/question | Avg input tokens |
|---|---:|---:|---:|---:|---:|
| api (`mistral-large-2407`) | 18/18 | 22.6 s | 1.9 s – 62.6 s | €0.1034 | 57,692 |
| local (`mistral:7b-instruct`) | **3/18** | 401 s (6.7 min) | 303 s – 548 s | €0.0000 | 7,400 |

**The local route only completed 3 of the 18 real questions within the backend's 600-second timeout** — the 2 questions over the smallest document (`conseil_etat_google_2020`, ~45k characters / ~11k tokens) and 1 of 2 attempts over the next-smallest (`edpb_guidelines_04_2021`, ~41k characters). Every question requiring a larger document (the CNIL sanction at ~200k characters, Schrems II at ~213k, VINCI at ~262k) timed out on this machine — a CPU-only host running `mistral:7b-instruct` at its real 32,768-token context window, sized to the actual prompt after fixing a bug where Ollama was silently truncating every prompt to 2,048 tokens regardless of the model's real window (see `src/backends.py`).

**What this means for the cost model:** the naive "sovereignty gate → free local route → near-zero cost" story does not survive contact with real document sizes on modest hardware. For the 15/18 questions where local timed out, the only way to get an answer at all was the API route — meaning, honestly stated, **local-only self-hosting on CPU-class hardware is not currently viable for this document population**; it works for short documents and fails outright, not gracefully, on long ones.

**Volume assumption and sensitivity (API route, since that's what's actually usable on documents this size on this hardware):**

- At the measured average of €0.1034/question and assuming local truly handles nothing above the size where it already fails (~50k characters), a monthly volume of **1,000 questions** on documents of this profile costs **≈ €103/month** at 100% API.
- **10× volume (10,000 questions/month):** ≈ €1,034/month — cost scales linearly with volume, since there is no fixed infrastructure cost in the API route itself (Mistral Large pricing has no volume discount tier visible via OpenRouter as of this writing).
- **API pricing doubles:** ≈ €207/month at the 1,000-question baseline — a real risk to flag, since OpenRouter/Mistral pricing is not contractually fixed for this POC.
- **If local were made viable** (larger local context via a bigger/quantized model, GPU acceleration, or a document-chunking strategy instead of single-shot full-context prompts) for the same 1,000 questions/month, the marginal API cost would approach €0 for whatever share moves to local — but that infrastructure investment (GPU instance, or engineering time for chunking) is not itself free and is out of scope for this POC's cost model.

**Also not counted above, and a real gap in "local is free":** the €0.0000 local cost figure only reflects that this POC ran the local model on hardware the author already owned, with no per-token billing. It excludes electricity, hardware amortization/depreciation, and — measured directly here — the **~17× latency penalty** (6.7 min vs 22.6 s) that would need to be priced into user experience or into provisioning enough local capacity to serve concurrent users at all.

## 6. Answer quality by route

Blind-graded 1–3 scale (1 = wrong/unusable, 2 = partially correct, 3 = correct and complete) against the `expected_answer_notes` planted in `eval/questions.csv`, graded without seeing which route produced which answer (`eval/blind_grade.py` / `eval/merge_grades.py`):

| Route | n | Avg score |
|---|---:|---:|
| api | 18 | 2.89 |
| local | 3 | 3.00 |

| Complexity | Route | n | Avg score |
|---|---|---:|---:|
| factual_lookup | api | 6 | 3.00 |
| factual_lookup | local | 2 | 3.00 |
| section_summary | api | 5 | 2.80 |
| section_summary | local | 1 | 3.00 |
| multi_hop | api | 4 | 3.00 |
| legal_interpretation | api | 3 | 2.67 |

**Read this carefully, not at face value:** local's n=3 is far too small to support "local ties or beats API on quality" as a general claim — it only reflects the 3 easiest, shortest documents that happened to complete at all (§5). The real, defensible finding here is about *reliability* (completion rate), not a quality edge for local.

On the API side, where the sample is large enough to say something, quality was high overall (2.89/3) with the softest spot on `legal_interpretation` (2.67/3). Both API answers scored 2 shared the same failure pattern: a long, well-structured, factually-consistent-with-the-document answer that nonetheless omitted one specific figure the grading note flagged as important (a financial-capacity figure in one case, a pole-level aggregate percentage in the other) — a "plausible but incomplete" failure, not a hallucination. This is a real, if narrow, limitation of long-context summarization even on the more capable model, worth flagging to anyone using this for a use case where a missed figure has consequences.

## 7. Limitations and honest caveats

- Sample sizes throughout are small (30 synthetic documents for the anonymizer gold set, 18 real questions × 2 routes attempted for the router/cost measurement) — enough to demonstrate the method and surface real failure modes, not enough to claim statistical significance. Every ratio above states its denominator.
- **The local route's real, measured limitation is not graceful degradation but outright failure on large documents**: 15 of 18 real questions timed out at 600 seconds rather than returning a lower-quality answer. This is the single most important number in this business case for anyone considering an all-local deployment: it works, cheaply, on short documents, and does not work at all on long ones without either a bigger local model, GPU acceleration, or a chunking strategy this POC does not implement.
- The anonymizer is evaluated on synthetic gold-set documents (Faker-generated FR data), not on the real corpus, because the real corpus's PII (if any) has no ground-truth annotation and must never be published — see `claude.md`. Its precision/recall numbers should be read as "measured on documents designed to contain hard cases", not as a guarantee on arbitrary real documents.
- This measurement run was itself constrained by the development machine's available memory (shared with several unrelated services) — the local-route sample size (3/18) reflects both the model's genuine context-window/timeout limitation and this machine's capacity, not a hard ceiling on what `mistral:7b-instruct` can do on adequately provisioned hardware. A dedicated GPU host would very likely close much of this gap; that was out of scope (and budget) for this POC.
