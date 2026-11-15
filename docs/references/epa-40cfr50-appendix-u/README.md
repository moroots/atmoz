# EPA ozone NAAQS data handling: 40 CFR Part 50, Appendix U

The authoritative rules for computing 8-hour averages and the daily maximum 8-hour average (MDA8) for ozone.
This is the spec for the MDA8 function planned for `atmoz.surface` utilities.

| File | What it is |
|---|---|
| `appendix-u.xml` | Raw text from the eCFR API (version dated 2026-11-01). Source of truth. |
| `appendix-u.md` | Readable version. Every paragraph is tagged with its citation, e.g. `**[3(b)]**`. |
| `appendix-u.sections.json` | Same content as `{sections: [{heading, paragraphs: [{id, text}]}]}` for programmatic lookup. |
| `appendix-u.versions.json` | eCFR version history of the appendix (last substantive change 2016-11-17, from 80 FR 65292). |

Search it with `grep -n "truncated" appendix-u.md` or by citation: `grep -n "^\*\*\[3(d)\]" appendix-u.md`.

Source: <https://www.ecfr.gov/current/title-40/chapter-I/subchapter-C/part-50/appendix-Appendix%20U%20to%20Part%2050>
(the eCFR is not an official legal edition of the CFR). Regenerate from the API:
`https://www.ecfr.gov/api/versioner/v1/full/<date>/title-40.xml?part=50&appendix=Appendix%20U%20to%20Part%2050`
(send `Accept-Encoding`, e.g. `curl --compressed`).

## The MDA8 rules in one place

| Rule | Cite |
|---|---|
| Hourly values in **ppm**, truncated (not rounded) to 3 decimals; hours are **local standard time** | 3(a) |
| 8-hour average stored at its **start hour**; valid with at least 6 of 8 hours; with 6 or 7 hours the divisor is 6 or 7 | 3(b) |
| With 5 or fewer hours, still valid if the sum of available hours is > 0.567 ppm (equivalent to > 0.070 with zeros substituted) | 3(b) |
| 8-hour averages truncated to 3 decimals | 3(b) |
| MDA8 = highest of the **17** averages starting 07:00 to 23:00 local (the last spans into the next day) | 3(c) |
| A day is valid with at least **13 of 17** valid averages, or if the MDA8 exceeds 0.070 ppm | 3(d) |
| Exceptional-event hours count as missing for averages (but count toward completeness in 3(d)) | 3(b), 3(d) |
| Collocated monitors: one primary, secondary values fill gaps (averaged if several) | 2(b) |
| Level of the standard: 0.070 ppm | 4(a) |
