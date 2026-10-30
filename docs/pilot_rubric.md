# Pilot labelling rubric (DRAFT, for Kareem and Yasmina)

Sheet: `data/llm/pilot/pilot_sheet_blank.csv` (150 rows). Label each row **independently** (do not compare until both are done) in the column `label_agree_neutral_disagree` with exactly one of `agree`, `neutral`, `disagree`. Then save as `labels_<name>.csv` with columns `pilot_id,label`.

**The question.** What is the *reply's stance toward the position taken in the parent comment*? Judge only the two texts shown. Ignore who wrote them, which side you think they are on, and whether you personally agree with either comment. Tone is not stance: a polite reply can disagree, a rude reply can agree.

## agree
The reply endorses, supports, confirms or extends the parent's position.

1. Parent: "Raising the minimum wage will cost jobs in rural areas." / Reply: "Yes, and the CBO numbers say the same thing. My town already lost two stores."
2. Parent: "Both parties take money from the same donors." / Reply: "Exactly, which is why neither will touch campaign finance."

## neutral
No clear stance toward the parent's position: a question, a request for a source, an unrelated remark, a joke that takes no side, or a genuinely mixed reply that supports and opposes the parent about equally.

1. Parent: "The poll shows her ahead by six points." / Reply: "Which poll is this? Link?"
2. Parent: "He should drop out after last night." / Reply: "Anyone know when the next debate is?"

## disagree
The reply rejects, contradicts, corrects or argues against the parent's position, whether politely or rudely.

1. Parent: "Voter fraud decided the primary." / Reply: "There is no evidence of that. Every audit found the counts matched."
2. Parent: "Trump will drop out before the convention." / Reply: "lol no. He has won more delegates than anyone."

## Rules for the hard cases
- **Sarcasm:** label the intended meaning ("Oh sure, because that always works" replying to a plan = disagree).
- **Partial agreement:** label the reply's stance on the parent's *main* point. If it genuinely splits evenly, use neutral.
- **Questions:** a rhetorical question that clearly implies a stance takes that stance; a real request for information is neutral.
- **Off-topic or insult-only replies** that say nothing about the parent's position: neutral. An insult that clearly rejects the parent's claim ("you're an idiot, that's false"): disagree.
- **Quotes:** if the reply quotes the parent and then argues, judge the argument, not the quote.
- **Cannot tell:** use neutral and add a note in an optional `note` column; we will count how often it happens.

Targets: report Cohen's kappa between raters, then resolve disagreements by discussion only *after* kappa is computed. The consensus label set is the agreed rows.
(The examples above are invented, not taken from the data.)
