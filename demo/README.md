# Demo

Screenshots and a recording of LakehouseIQ running on Databricks Free Edition. The live app needs a sign-in to
the workspace it runs in, so these are the way to see it without one.

| File | Shows |
| --- | --- |
| `01-answer.png` | "Top 5 products by net revenue?" answered from the `mv_order_lines` metric view, with a summary |
| `02-sql-follow-up.png` | Follow-up "Only Electronics": the generated SQL keeps the context and adds the filter, using `MEASURE()` |
| `03-quality.png` | The verdict: relevance and SQL quality from the judge model, schema fit and value checks from code |
| `recording.mp4` | *(to add)* a 2-3 minute walkthrough |

## Recording checklist

Aim for 2-3 minutes; keep the file under ~25 MB.

1. Ask "Top 5 products by net revenue?"; open the **SQL**, **Quality** and **Trace** tabs.
2. Ask the follow-up "Only Electronics" to show conversation context.
3. Ask a harder one, e.g. "Which departments are more than 10% over budget year-to-date?".
4. Show the benchmark table in the main README (accuracy, silent-wrong rate and latency vs Genie).

To play inline on the main README page, drag the video into the README editor on github.com: GitHub hosts it and
inserts a player link. A video committed to the repo shows as a download link instead.
