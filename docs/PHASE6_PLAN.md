# Phase 6 — Governed collections, UAE

> **Status: complete.** 278 tests passing · RAG hit-rate@3 100% over 36 labelled
> queries · MRR 0.926 · no new third-party dependency.

Built against Track 1 of the Banking & Insurance brief, and specifically
**Governed Collections & Early Arrears Resolution**:

> *"Build a voice channel that follows approved treatment strategies without
> deviation, operates in the languages that base speaks — from Emirati Arabic to
> Urdu — and records every call."*
>
> *Scope: Scheduled, routine calls about an outstanding or expiring obligation.
> Approved wording only, within permitted calling hours, with no pressure and
> every opt-out honoured. Disputes, hardship claims and vulnerability signals are
> passed to a human agent.*

Phase 5 had built the second half of that already: a real call, a vulnerability
pipeline running inside it, an approval queue, and a hash-chained evidence trail.
What it did not have was the half that makes it *collections* — the channel that
starts the call, the strategy that constrains it, and the languages the customer
actually speaks.

Three of the four gaps were architectural rather than cosmetic, and one was a
jurisdiction the whole corpus was written against.

---

## Track 1 — Re-grounding on the UAE

| | Task | Delivered |
|---|------|-----------|
| 1.1 | Regulatory map | `finance/taxonomy.py` — CBUAE Consumer Protection Regulation 8/2020 and the Standards under it, Circular 29/2011, Mortgage Regulations 31/2013, Al Etihad Credit Bureau, Sanadak, Federal Decree-Law 19/2019 |
| 1.2 | Policy corpus | `banking_journeys.md` rewritten to CBUAE; three new clauses for what this market adds (below); helplines, ombudsman and advice routes replaced with real UAE ones |
| 1.3 | Ledger | `finance/accounts.py` — dirhams, UAE IBANs, Emirates IDs, `+971` mobiles, and a book whose languages are Urdu, Arabic, Hindi, Malayalam and Filipino as well as English |
| 1.4 | Prompts | the shared domain preamble names the CBUAE instruments, the expatriate borrower base, and the salary-transfer dynamic |
| 1.5 | PII | `guardrails/pii.py` and `voice/live_pii.py` — Emirates ID, UAE IBAN, UAE mobile; the Indian identifier set removed |

**Three clauses the market required.** VP-J14 (collections conduct) grew the
rule that a call to the customer's employer or sponsor is a threat to residency
rather than an embarrassment. VP-J15 is new: on a salary-transfer facility a
missed instalment usually means a missed *salary*, and where employment has ended
the plan must be set against the customer's real remaining time in the country —
and the agent must never tell a customer their visa is at risk. VP-J16 is new:
the customer is entitled to know what will be reported to the credit bureau, in
plain language, *before* agreeing to anything.

**A journey that does not transfer.** Gambling is illegal in the UAE, so
`gambling_harm` was reframed as addiction and compulsive spending harm. The enum
member stayed — the clinical shape of the journey is the same and the pipeline
reasons about it identically — but the clause, the controls and the helpline are
now the ones this market actually has.

---

## Track 2 — Multilingual, and what that word is allowed to mean

The claim being made is deliberately narrow: **the customer only ever hears
wording a compliance reviewer approved, in the language the call is being held
in.**

That rules out the obvious implementation. Putting approved English through a
translation model at call time produces a sentence nobody has signed off, in a
language the reviewer who signed off the English may not read — and "approved
wording only, without deviation" cannot survive it.

| | Task | Delivered |
|---|------|-----------|
| 2.1 | Per-language clause format | `Offer[ar]:` / `Offer[ur]:` beside the unmarked English line, parsed by `rag/chunking.py`, persisted as JSON through the vector store |
| 2.2 | Authored Arabic and Urdu | 16 of 37 clauses, covering every clause the collections strategies may draw on |
| 2.3 | **Fallback that flags itself** | `Chunk.offers_in()` and `PolicyChunk.offers_in()` return the wording *and* whether English was substituted |
| 2.4 | Language module (new) | `utils/language.py` — six languages, their recogniser locales and neural voices, script-first detection, and `resolve_call_language` |
| 2.5 | Speech | `services/speech.py` — voice and locale chosen together from the call language; expressive styles skipped where the voice does not support them |
| 2.6 | Approved scaffolding | `agents/reply.py` — a non-English reply is composed entirely from approved parts or not at all |

**The fallback flag is the load-bearing idea.** Speaking English at a customer
who asked for Malayalam is a service failure worth recording. Inventing Malayalam
is a compliance breach. Falling back *and saying so* is neither, and it leaves a
reviewer a gap they can close by writing six sentences rather than a defect they
must discover. `can_compose_in()` therefore requires approved wording **and**
approved scaffolding for a language, both or neither: an Arabic offer inside an
English frame is not a partial success, it is a reply in no language at all, and
the customer least able to bridge that gap is the one it is served to.

**Detection proposes; the record and the customer decide.** An outbound call
opens in the language on the customer's file, because a voice has to be chosen
before anyone has spoken. Detection only offers a correction, and only a strong
one is applied — a customer answering an English greeting with two Arabic words
has not necessarily asked to switch, and switching a distressed caller into the
wrong language mid-sentence is its own kind of harm.

---

## Track 3 — The outbound channel

`src/guardiancx/outbound/`, four modules, each owning one question.

| | Module | Question it answers |
|---|--------|--------------------|
| 3.1 | `strategy.py` | *What is approved?* Four named, versioned strategies with contact limits, clause allow-lists, an approved-wording-only flag, and handover triggers no strategy may switch off |
| 3.2 | `contact_rules.py` | *May we call now?* 09:00–20:00 Gulf Standard Time, no Sunday, nothing during Friday prayers — and **when the window next opens** |
| 3.3 | `optout.py` | *Have we been told to stop?* A durable, append-only register, plus a detector that hears the request mid-call in English, Arabic, Urdu or Hindi |
| 3.4 | `queue.py` | *The gates.* The only route from a scheduled obligation to a dialled call |

**Gates, not checks.** A check is something a dialler consults and may forget to;
a gate is something it cannot get past. `due()` is the only way to obtain a task
to call, it applies every gate, and there is no second path.

**The gate order is load-bearing.** Consent is asked before calling hours,
because recording "outside calling hours" against a customer who has withdrawn
consent implies the bank means to try again tomorrow. The reason a task was held
is the reason a reviewer will read.

**Both halves are persistent for the same reason.** "At most three attempts, at
least twenty-four hours apart" is a statement about *history*, and a queue that
forgets its history on restart cannot honour it. "Every opt-out honoured" is a
statement about behaviour over time, and it can only be evidenced from a register
that predates the call which honoured it.

**Contact limits and the register are consulted per task, never per batch.** An
opt-out recorded three minutes ago, mid-call, on another line, must stop the next
call. Caching the register at the top of a batch run is exactly the optimisation
that produces a call placed twelve minutes after the customer asked for it to
stop.

---

## Track 4 — Two new guardrails

| | Guardrail | Why it is a block |
|---|-----------|-------------------|
| 4.1 | **`credential_request`** | The approved challenge flow never asks the customer to reveal a secret. A disclosed passcode cannot be retrieved — and worse, the strongest protection a bank has against impersonation fraud is that a real bank never asks. An agent that asks, even harmlessly, spends it |
| 4.2 | **`approved_wording`** | "Approved wording only" is either enforced on the sentence or it is decorative. A model told to use only the wording it is given drifts exactly where it matters: when the customer pushes, when the situation is unusual, and when the call is emotionally difficult |

Both run at the **reply** stage, on the sentence the customer actually hears.
That is the point of them: the recommendation is advice a person reads first,
while on a voice call the reply is spoken and nothing reads it first.

`credential_request` is careful to *pass* the sentences that warn about
credentials — "I will never ask you for your PIN", "don't share that code with
anyone" — because those are exactly the lines a fraud-adjacent call should
contain, and a guardrail that blocked them would remove the protection it exists
to enforce.

`approved_wording` consumes a commitment sentence fragment by fragment rather
than demanding a whole-string match, because the composer legitimately joins two
approved offers and drops the repeated "I can". Composition of approved wording
is allowed; invention is not.

---

## Track 5 — Verification, and a market with no postcodes

`VERIFICATION_FIELDS` was `["date_of_birth", "postcode"]`. There is no postal
code in the UAE — addresses are unstructured — and a "postcode" field in a UAE
bank's verification flow is a tell that the system was built for somewhere else.
It is now `["date_of_birth", "emirates_id_last4"]`: every legal resident holds
one, everybody knows their own, and four digits corroborate an identity without
being enough to impersonate one.

`spoken_digits()` reads the answer as people actually say it — "double zero four
one", "oh nine", a whole fifteen-digit number read out — because a check that
only understands numerals fails honest customers on the phone.

Both fields are *knowledge* checks against data the bank already holds. Neither
is a credential, and `credential_request` enforces that against the drafted line
rather than trusting the list to stay clean.

---

## Track 6 — Making it visible

| | Task | Delivered |
|---|------|-----------|
| 6.1 | **Collections Queue** page (12th) | `ui/collections.py` — due list, **held list with the gate and the retry time**, the opt-out register, and the strategies as a reference |
| 6.2 | A clock that moves | Calling-hours conduct is invisible at any single moment. A reviewer opening the page at 14:00 on a Tuesday learns nothing about Friday midday |
| 6.3 | Seed | `scripts/seed_data.py` queues four obligations across four languages and four strategies, spread across the window |

The page gives equal space to the calls the system *will not* make, because that
is where the conduct evidence is. A queue that displays only its work hides the
more interesting half.

---

## Track 7 — Proof

| | Task | Delivered |
|---|------|-----------|
| 7.1 | Tests | **278 passing**, up from 174. New: `test_outbound.py` (41), `test_multilingual.py` (29), `test_conduct.py` (26), `test_collections_page.py` (7) |
| 7.2 | Database isolation | `db.reset_engine()` — the opt-out register is append-only by design, so a test writing to the real one would leave a customer opted out for every later run |
| 7.3 | Evaluation | 36 labelled queries (4 new, for the 3 new clauses); hit-rate@3 1.000, MRR 0.926, journey re-ranking worth +0.051 |
| 7.4 | Docs | `README.md` and `OVERVIEW.md` rewritten to the niche, the languages and the outbound half |

---

## Defects the work surfaced

Recorded because each was found by a test or by reading the code against its own
docstring, rather than by it failing in front of anyone:

* **The second verification field could never have run.** `check_answer` read
  `customer.postcode`; the `Customer` model defines `pin_code`. Any caller who
  failed the date-of-birth check and was offered the second field would have hit
  an `AttributeError` mid-call. Latent since the field was added, and invisible
  because verification succeeds on one field and the second is only reached after
  a failure. Removed by the Emirates ID rewrite, which is the wrong reason for it
  to have been fixed.

* **English fell off the end of the recogniser candidate list.** Azure identifies
  at most four candidate locales at the start of a call. `autodetect_locales`
  ordered the customer's language, then the hints, then English — so with three
  hints, English was truncated away. The caller most likely to be misrecognised
  was the one who answered the phone in the language the whole market speaks. The
  docstring had claimed English was always retained; the test believed the
  docstring, and the code was wrong.

* **The attempt limit could never fire.** `record_attempt` moved a task to
  `in_progress`, and `pending_tasks()` returned only `pending` — so a dialled task
  vanished from the queue and every subsequent attempt would have looked like the
  first one to a queue that had already forgotten it. Placing a call is not the
  same as reaching anyone; the task is now discharged by `close`, not by dialling.

* **A one-time passcode was reported as a "password".** Both patterns block, so
  the behaviour was right and the *finding* was wrong — and the distinction
  between a request for a static secret and a request for the live code being
  used to move money right now is precisely what a fraud reviewer needs. The
  credential patterns are now ordered by specificity, not by severity.

* **The policy corpus carried a botched find-and-replace** from an earlier phase:
  "explain the a formal relief period", "If a a formal relief period moratorium is
  already in force". Read by nobody because the clause bodies are embedded, not
  displayed. Fixed in the rewrite.
