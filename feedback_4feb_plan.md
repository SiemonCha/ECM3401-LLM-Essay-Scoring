Got it. Here's the priority order based on the feedback:

**1. Krippendorff's Alpha** (measure how much 3 prompts agree with each other)
Install the `krippendorff` Python package, treat each prompt variant as an annotator per essay, compute alpha per model×strategy. Plug results into a table in Section 4.1.5.

**2. Fix correlation method** (instead of 3 groups short/medium/long, use all individual)
Replace the three-category correlation (r=-0.424) with a per-essay Pearson correlation using raw word count. You already have `word_count` in your CSVs — just recompute.

**3. TTR + Lexical Density** (check quality not just length; TTR = vocab length, lexical dense = information-dense)
Requires the actual essay text (your `sample_100.csv`). Use spaCy to tokenise, compute TTR (unique lemmas / total tokens) and lexical density (content POS / total tokens) per essay. Then correlate with accuracy and robustness.

**4. Merge Figures 5 and 6**
Instead of two separate subplots (accuracy left, robustness right), produce one plot per phase where the line = accuracy and error bars = SD. Supervisor explicitly asked for this.

**5. Parse Tree Depth** (measure syntactic complexity/ various grammar)
Needs the Stanza library which is large. Compute mean constituency tree depth per sentence per essay. Supervisor said "if you have time" — do this last.
