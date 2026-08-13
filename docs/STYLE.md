# Documentation voice and page contract

**Audience:** Contributor

Use this page when you add or edit user-facing docs. The reader of those pages
may be new to programming *and* new to markets. Write so they can finish a
safe first task without a quant background.

**Related:** [Docs index](README.md) · [Glossary](GLOSSARY.md) · [Sample trade](SAMPLE-TRADE.md)

---

## Voice

- **Helpful.** Open with what the page is for, then the smallest next step.
  Explain *why* a command exists before listing flags.
- **Friendly.** Second person, short sentences, no condescension. It is fine
  to say “this term looks dense — here is the plain idea.”
- **Jargon-aware.** On first use of a market or systems word, add a one-line
  gloss and link the [glossary](GLOSSARY.md). Never assume pip, lot, spread,
  horizon, MCP, or TOON are known.
- **Progressive disclosure.** Lead with the 80% path: demo account, read-only
  tools, a simple forecast (Theta), Web UI or one CLI command. Put method
  internals, clock notes, and payload contracts under **Deeper detail**.
- **Calm safety.** Demo-first and dry-run-first. Be direct about live orders
  without scare language.

## Audience tags

Every index row and new page should carry one tag:

| Tag | Who | Examples |
|-----|-----|----------|
| **User** | Someone using mtdata to look at markets | [WEBUI.md](WEBUI.md), [SAMPLE-TRADE.md](SAMPLE-TRADE.md) |
| **Operator** | Someone hosting, configuring, or scripting | [ENV_VARS.md](ENV_VARS.md), [DEPLOYMENT.md](DEPLOYMENT.md), [OUTPUT.md](OUTPUT.md) |
| **Contributor** | Someone changing the product | this page, [WEBUI_GOAL.md](WEBUI_GOAL.md), [DEPENDENCY_MIGRATION.md](DEPENDENCY_MIGRATION.md) |

Do **not** rewrite Operator or Contributor references into beginner prose.
Add a 3–5 line plain-English lead and a bounce link to a User page.

## Page contract (User pages)

1. Title states the job (“Look at the chart”), not the module name.
2. The opening paragraph is plain English. No acronym unless it is glossed
   in the same sentence.
3. Include **Related** links. If the page uses specialized vocabulary, add a
   **Dense terms** line that points at glossary anchors.
4. The first copy-paste example is **read-only** unless the page is explicitly
   about live orders.
5. Put optional extras, algorithm lists, and envelope fields after
   **Deeper detail**.

## First-screen list — do not leave unexplained

If any of these appear in the first screen of a User page, gloss them or move
them down: BOCPD, Kelly, CVaR, HMM, TOON, MCP, NDJSON, Session 0, markout,
implementation shortfall, SAX, PAA, Heston, ADF, KPSS.

## Checklist for a docs PR

- [ ] Audience tag is visible from the [index](README.md).
- [ ] First 15 lines of a User page have no unexplained acronyms.
- [ ] New tools are linked from a User page, not only a CLI table cell.
- [ ] Safety claims about Web API / Tools runner match [WEB_API.md](WEB_API.md)
      and [TRADING_SAFETY.md](TRADING_SAFETY.md).
- [ ] Credential examples stay placeholders (never real logins).
