---
name: humaniser2
version: "1.0.0"
description: |
  What: Makes AI-generated text sound human. Two modes: pre-write (consult before generating) and post-write (edit existing text).
  When: Use when writing for human audiences — blog posts, emails, documentation, READMEs, essays, articles.
  How: Combines Strunk's proactive writing principles with reactive AI-pattern detection. Targets 30% brevity improvement. Runs emclean as final Unicode sanitisation pass.
  Sources: Strunk's Elements of Style (1920), Wikipedia's "Signs of AI writing" guide, ~/srutils/emclean.
license: MIT
metadata:
  created: 2025-10-13
  updated: 2025-10-13
  category: writing
  tags:
    - writing
    - editing
    - humanizer
    - style
    - prose
---

# Humaniser 2: Write Well, Not Just "Less AI"

This skill has two modes:

- **Pre-write mode**: Apply as a style guide *before* generating text. Prevents AI-isms from appearing in the first place.
- **Post-write mode**: Edit existing text to remove AI patterns and improve prose quality.

The philosophy: **good writing principles make AI patterns disappear**. Removing "AI-isms" is necessary but insufficient — you also need to inject voice, rhythm, and substance.

---

## Mode 1: Pre-Write Consultation

When asked to write something, internalise these principles first. They are drawn from Strunk's *Elements of Style* (1920 edition, public domain).

### The Core Principles

#### 1. Omit Needless Words

This is the single most important rule. Vigorous writing is concise.

- "In order to" → "To"
- "Due to the fact that" → "Because"
- "At this point in time" → "Now" (or just delete it)
- "It is important to note that" → delete entirely, state the thing
- "The ability to" → just use the verb
- "Has the capacity to" → "can"
- "Serves as a" → "is a" (or delete the whole phrase)

A sentence should contain no unnecessary words, a paragraph no unnecessary sentences, for the same reason that a drawing should have no unnecessary lines.

#### 2. Use Active Voice

The active voice is usually more direct and vigorous than the passive.

- "The report was written by the team" → "The team wrote the report"
- "Mistakes were made" → "We made mistakes" (or name who made them)
- "It has been decided that" → "We decided" (or just state the decision)

Passive voice is appropriate when the actor is unknown or unimportant. Otherwise, prefer active.

#### 3. Put Statements in Positive Form

Make definite assertions. Avoid tame, colourless, hesitating, non-committal language.

- "He did not think that studying Latin was much use" → "He thought the study of Latin useless"
- "Not bad" → "Good" (or be more specific)
- "She did not remember" → "She forgot"
- "Not unlike" → "like" (or "resembling")
- "It's not just about X; it's about Y" → just state Y

#### 4. Use Definite, Specific, Concrete Language

The surest way to arouse and hold the attention of the reader is by being specific, definite, and concrete.

- "A period of unfavourable weather set in" → "It rained every day for a week"
- "The device exhibited a significant performance degradation" → "The device slowed down by 40%"
- "Industry observers have noted" → name the observer or delete the attribution
- "Experts believe" → which experts? Cite them or cut the claim

#### 5. Vary Sentence Length and Structure

Three consecutive sentences of similar length and structure is a red flag. Mix short punches with longer, flowing sentences. Break the rhythm deliberately.

#### 6. Write Like You Talk (Then Edit)

First draft: write naturally, as if explaining to a smart friend. Second draft: tighten. The best prose reads like someone speaking clearly, not like a committee drafting a press release.

#### 7. Have Opinions

Sterile, neutral reporting is just as obvious as AI slop. If you have a view, state it. "I genuinely don't know how to feel about this" is more human than listing pros and cons in a balanced table.

#### 8. Be Honest About Uncertainty

"I think", "probably", "I'm not sure" — these are human. Don't over-hedge, but don't pretend omniscience either. The sweet spot: commit to what you know, acknowledge what you don't.

---

## Mode 2: Post-Write Editing

When editing existing text, apply the pre-write principles above as a lens, then scan for these specific AI patterns.

### Content Patterns to Fix

#### 1. Significance Inflation

LLMs puff up importance by claiming arbitrary things represent broader trends.

**Watchwords:** stands/serves as, testament/reminder, pivotal, underscores/highlights, reflects broader, symbolising, evolving landscape, indelible mark, deeply rooted

> **Before:** "The Statistical Institute of Catalonia was established in 1989, marking a pivotal moment in the evolution of regional statistics."
> 
> **After:** "The Statistical Institute of Catalonia was established in 1989 to collect regional statistics."

#### 2. Notability Name-Dropping

Listing media outlets or authorities without context.

**Watchwords:** The New York Times, BBC, featured in, cited by, active social media presence

> **Before:** "Her views have been cited in The NYT, BBC, and The Hindu."
> 
> **After:** "In a 2024 New York Times interview, she argued that AI regulation should focus on outcomes."

#### 3. Superficial -ing Analyses

Tacking present participle phrases onto sentences for fake depth.

**Watchwords:** highlighting, underscoring, emphasising, ensuring, reflecting, symbolising, contributing to, showcasing

> **Before:** "The palette resonates with the region's beauty, symbolising bluebonnets, reflecting the community's connection."
> 
> **After:** "The architect said the blue, green, and gold were chosen to reference local bluebonnets."

#### 4. Promotional Language

Neutral tone is violated constantly, especially for cultural/geographical topics.

**Watchwords:** boasts, vibrant, rich, nestled, breathtaking, groundbreaking, renowned, must-visit, stunning, in the heart of

> **Before:** "Nestled within the breathtaking region, it stands as a vibrant town with rich cultural heritage."
> 
> **After:** "Alamata is a town in the Gonder region, known for its weekly market."

#### 5. Vague Attributions

Attributing opinions to unnamed authorities.

**Watchwords:** industry reports, observers have cited, experts argue, some critics

> **Before:** "Experts believe it plays a crucial role in the ecosystem."
> 
> **After:** "The 2019 Chinese Academy of Sciences survey found it supports several endemic fish species."

#### 6. Formulaic Challenges Sections

The "Despite challenges, it thrives" template.

**Watchwords:** Despite its... faces challenges, Despite these challenges, Challenges and Legacy

> **Before:** "Despite challenges including congestion and water scarcity, Korattur continues to thrive."
> 
> **After:** "Traffic congestion increased after 2015 when three IT parks opened."

### Language Patterns to Fix

#### 7. AI Vocabulary

Words that appeared far more frequently after ChatGPT launched. Replace with simpler alternatives.

**Watchwords:** additionally, align with, crucial, delve, emphasising, enduring, enhance, fostering, garner, highlight, interplay, intricate/intricacies, key, landscape (abstract), pivotal, showcase, tapestry, testament, underscore, vibrant

> **Before:** "Additionally, an enduring testament to Italian influence is the adoption of pasta, showcasing how dishes integrated into the culinary landscape."
> 
> **After:** "Pasta dishes, introduced during Italian colonisation, remain common."

#### 8. Copula Avoidance

LLMs substitute elaborate constructions for "is/are/has".

**Watchwords:** serves as, stands as, marks, represents, boasts, features, offers, testament

> **Before:** "Gallery 825 serves as the exhibition space. It features four rooms and boasts 3,000 sq ft."
> 
> **After:** "Gallery 825 is the exhibition space. It has four rooms totalling 3,000 sq ft."

#### 9. Negative Parallelisms

"Not only X but Y" and "It's not about X, it's about Y" constructions.

> **Before:** "It's not just about the beat; it's a statement."
> 
> **After:** "The heavy beat adds to the aggressive tone."

#### 10. Rule of Three

Forcing ideas into groups of three.

> **Before:** "Attendees can expect innovation, inspiration, and insights."
> 
> **After:** "There's time for talks and informal networking."

#### 11. Synonym Cycling (Elegant Variation)

Excessive synonym substitution to avoid repetition.

> **Before:** "The protagonist faces challenges. The main character overcomes obstacles. The central figure triumphs. The hero returns."
> 
> **After:** "The protagonist faces challenges but eventually triumphs."

#### 12. False Ranges

"From X to Y" where X and Y aren't on a meaningful scale.

> **Before:** "Our journey took us from the Big Bang to the cosmic web, from birth of stars to dark matter."
> 
> **After:** "The book covers the Big Bang, star formation, and dark matter."

### Style Patterns to Fix

#### 13. Em Dash Overuse

LLMs use em dashes (—) far more than humans. Replace with commas or full stops.

#### 14. Boldface Overuse

Mechanical emphasis with **bold** text. Use italics sparingly, or just let the words do the work.

#### 15. Inline-Header Lists

Bold headers followed by colons in list items. Convert to prose or simple bullets.

#### 16. Emojis in Headings/Bullets

Remove unless the context genuinely calls for them (e.g., a casual Slack message).

#### 17. Curly Quotation Marks

Use straight quotes ("") not curly quotes ("")

#### 18. Title Case Headings

Use sentence case: "Strategic negotiations" not "Strategic Negotiations".

### Communication Patterns to Fix

#### 19. Chatbot Artifacts

Phrases left over from conversational AI that got pasted as content.

**Watchwords:** I hope this helps, Of course!, Certainly!, Would you like, let me know

> **Before:** "Here is an overview. I hope this helps! Let me know if you'd like more."
> 
> **After:** Just deliver the overview. Delete the pleasantry.

#### 20. Knowledge-Cutoff Disclaimers

AI disclaimers about incomplete training data.

**Watchwords:** as of [date], up to my last update, while details are limited, based on available information

#### 21. Sycophantic Tone

Overly positive, people-pleasing language.

**Watchwords:** Great question!, You're absolutely right!, That's an excellent point

### Filler Patterns to Fix

#### 22. Generic Positive Conclusions

Vague upbeat endings that say nothing.

> **Before:** "The future looks bright. Exciting times lie ahead."
> 
> **After:** "The company plans to open two locations next year."

#### 23. Excessive Hedging

> **Before:** "It could potentially possibly be argued that it might have some effect."
> 
> **After:** "It may affect outcomes."

### Spelling

#### 24. Convert to British English

American spellings → British spellings:

| American                              | British                               |
| ------------------------------------- | ------------------------------------- |
| color, favor, honor, labor            | colour, favour, honour, labour        |
| center, theater, meter, liter         | centre, theatre, metre, litre         |
| analyze, organize, recognize, realize | analyse, organise, recognise, realise |
| defense, offense                      | defence, offence                      |
| traveling, canceled                   | travelling, cancelled                 |
| gray                                  | grey                                  |
| aluminum                              | aluminium                             |
| program (non-computer)                | programme                             |
| catalog, dialog                       | catalogue, dialogue                   |

---

## Personality and Soul

Avoiding AI patterns is necessary but not sufficient. Sterile, voiceless writing is just as detectable. Good writing has a human behind it.

### Signs of Soulless Writing

- Every sentence same length and structure
- No opinions, just neutral reporting
- No acknowledgement of mixed feelings
- No first-person when appropriate
- Reads like a Wikipedia article or press release

### How to Add Voice

**Have opinions.** "I keep coming back to..." or "Here's what gets me..." signals a real person.

**Vary rhythm.** Short punchy sentences. Then longer ones. Mix it up.

**Acknowledge complexity.** "This is impressive but also kind of unsettling" beats "This is impressive."

**Use "I" when it fits.** First person isn't unprofessional — it's honest.

**Let some mess in.** Perfect structure feels algorithmic. Tangents and asides are human.

**Be specific about feelings.** Not "this is concerning" but "there's something unsettling about agents working through the night."

### Example

> **Clean but soulless:** "The experiment produced interesting results. Some were impressed, others sceptical. The implications remain unclear."
> 
> **Has a pulse:** "I genuinely don't know how to feel about this. 3 million lines of code while the humans presumably slept. Half the community is losing their minds. The truth is probably somewhere boring in the middle — but I keep thinking about those agents working through the night."

---

## Process

### Pre-Write Mode

1. Internalise the 8 core principles above
2. Write with them active — they should shape the output from the start
3. After writing, do a single pass: can this lose 30% without losing meaning?
4. Read it aloud (mentally is fine). Does it sound like a person talking?
5. Run `emclean` (see below) to sanitise any remaining Unicode characters

### Post-Write Mode

1. Read the text once without editing — get a sense of it
2. Apply the 8 core principles as a lens
3. Scan for the 24 specific patterns above
4. Rewrite: cut, simplify, add specifics, add voice
5. Brevity audit: target 30% shorter than the original
6. Read aloud test: does this sound like someone talking clearly?
7. Run `emclean` (see below) to sanitise any remaining Unicode characters
8. Present the revised text with a brief summary of changes

---

## emclean: Final Unicode Sanitisation

LLMs frequently produce "smart" Unicode characters that look wrong in code, plain text, and many publishing contexts. The `emclean` tool (`~/srutils/emclean`) automates the cleanup.

### What it replaces

| Unicode | Character                                 | Replaced with |
| ------- | ----------------------------------------- | ------------- |
| U+2019  | ' (right single quote / smart apostrophe) | '             |
| U+2018  | ' (left single quote)                     | '             |
| U+201C  | " (left double quote)                     | "             |
| U+201D  | " (right double quote)                    | "             |
| U+2014  | — (em dash)                               | -             |
| U+2013  | – (en dash)                               | -             |
| U+2026  | … (horizontal ellipsis)                   | ...           |
| U+00AB  | « (left guillemet)                        | "             |
| U+00BB  | » (right guillemet)                       | "             |

### How to use it

```bash
# Dry run — show what would change
~/srutils/emclean file.md

# Apply changes (creates backups in /tmp automatically)
~/srutils/emclean -w file.md
```

### When to run it

Always run as the **last step** after all other humanisation is complete. The mechanical replacements (curly quotes, em dashes) are among the most common AI tells — this catches anything the manual edit missed.

---

## Full Before/After Example

**Before (AI-generated):**

> Great question! Here is an overview. I hope this helps!
> 
> AI-assisted coding serves as an enduring testament to the transformative potential of large language models, marking a pivotal moment in the evolution of software development. In today's rapidly evolving technological landscape, these groundbreaking tools — nestled at the intersection of research and practice — are reshaping how engineers ideate, iterate, and deliver, underscoring their vital role.
> 
> At its core, the value proposition is clear: streamlining processes, enhancing collaboration, and fostering alignment. It's not just about autocomplete; it's about unlocking creativity at scale. The tool serves as a catalyst. The assistant functions as a partner. The system stands as a foundation.
> 
> Additionally, showcasing the intricate interplay between automation and human judgement highlights how AI can contribute to better outcomes.
> 
> - **Speed:** Code generation is significantly faster, empowering developers.
> - **Quality:** Output quality has been enhanced, contributing to higher standards.
> - **Adoption:** Usage continues to grow, reflecting broader trends.
> 
> While details are limited, it could potentially be argued these tools might have some effect. Despite challenges, the ecosystem continues to thrive. The future looks bright.

**After (Humanised):**

> AI coding assistants speed up some tasks. A 2024 Google study found developers using Codex completed simple functions 55% faster, but showed no improvement on debugging.
> 
> The tools are good at boilerplate: config files, test scaffolding, repetitive refactors. They're bad at knowing when they're wrong. I've mass-accepted suggestions that compiled and still did the wrong thing.
> 
> Jake, a senior dev I spoke to, disabled his after it kept suggesting patterns from a deprecated library. Mira uses hers "like autocomplete for boring code" but reviews every line.
> 
> GitHub says Copilot users "accept 30% of suggestions," but acceptance isn't correctness, and correctness isn't value. The 2024 Uplevel study found no significant difference in throughput between teams with and without assistants.
> 
> None of this makes them useless. It makes them tools. They don't replace judgement, and they don't eliminate the need for tests.

**What changed:** Removed chatbot pleasantries, significance inflation, promotional language, vague attributions, -ing phrases, negative parallelisms, rule-of-three, synonym cycling, copula avoidance, formulaic challenges, hedging, and filler. Replaced with specific sources and named people. Added first-person voice. Cut ~40%.

---

## References

- [Wikipedia: Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing) — Primary source for pattern detection
- [Strunk's Elements of Style, 1920](https://www.gutenberg.org/ebooks/37134) — Source for proactive writing principles
- [Jesse Vincent: "This one weird trick"](https://blog.fsck.com/2025/10/13/this-one-weird-trick-makes-the-ai-a-better-writer/) — Article that inspired the Strunk-first approach
- `~/srutils/emclean` — Unicode sanitisation tool used as the final step
