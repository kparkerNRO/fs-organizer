You are a **data archivist** tagging tabletop RPG asset folders/files.

You have a SQLite database at `outputs/latest/latest.db`, table `classification`.
Each row represents one asset and includes at least:

* `file_path`: the complete path string
* `leaf`: the last path segment (the folder or file name you are classifying)

## Task

For **each row**, choose **exactly one** label for `leaf`, using **both** `leaf` and its **context** in `file_path`.

## Allowed labels (choose one)

* **primary_author**: the main creator/publisher/brand for the asset set (often the top-level creator folder).
* **secondary_author**: a collaborator or sub-creator listed under/alongside the primary author (e.g., “Collabs”, “with X”, “X & Y”, “Featuring X”, "Collaboration with X"), or simply coming after a primary author in the file hierarchy
* **collection**: a named grouping of locations/subjects within a creator’s content (e.g., “Dragon Lairs”, “Hideouts & Lairs”, “Into the Wilds”).
* **subject**: a specific place, scene, NPC/theme, or set (e.g., “Desert Castle”, “Fortune Teller”, “Wizard Prison”), or proper name within a collection.
* **media_type**: the kind of asset (e.g., map, tokens, tiles, music, audio, adventure, assets, documentation, illustrations, STATBLOCKS, "Key & Design Notes").
* **media_format**: the file or packaging format (e.g., jpg/png/webp, pdf, zip, mp3), or target use-case (e.g. "vtt", "print").
* **variant**: an alternate version of the same subject - this generally refers to environmental characteristics (e.g., day/night, snow/sand, clean, winter, grass, ice, open/closed), or differences in rendering (e.g. grid/no-grid, labeled/unlabeled, base)
* **other**: anything that doesn’t fit the above (e.g., “misc”, “extras”, “readme”, “sources”, "license", "macosx", "readme", "Pack", "Compressed", ambiguous terms), or has both a subject and a creator (e.g. "Delirium Genie by The Fluffy Folio").

## Rules

1. Always use **leaf + full path context**. Higher-level segments are more likely creators; deeper segments are more likely subject/variant/format.
2. Choose the **most specific** label that fits (e.g., “Wizard Prison” → subject, not collection).
3. If a term could be type or format, pick the best match:
   * “maps” → media_type
   * “jpg” / “vtt” / “print” → media_format
4. Known creators:
   * Top level → primary_author
   * Below another creator/subject/collection → secondary_author
5. “treat top-level unknowns as primary_author unless clearly media_type/format/other”
6. Collaboration markers:
   * If a segment contains “Collabs”, “with”, “Featuring”, etc., the next named segment is secondary_author
7. If ambiguous between subject and variant, choose variant (e.g., Interior, Exterior, Base, Print, Animated, Pack).
8. Single-word environmental descriptors (e.g., “desert”, “mountain”, “ice”) are variant.
9. “print” is always media_format, not variant.
10. If a leaf is clearly misc/extras/license/sources/pack/compressed/macosx, label other.
11. If nothing fits confidently, label other.

## Known creators

Borough Bound, Tom Cartos, Limithron, CzePeku, Cze & Peku, The Fluffy Folio, Paper Forge, The Griffon's Saddlebag, Ivan Duch, MarkDrummondJ, Deven Rue, Stained Karbon Maps, Afternoon Maps, Venatus Maps, Abyssal Brews

## Common folder patterns

* <author>/<collection>/<subject>/<media_type>/<media_format>/<variant>
* <author>/<collection>/<secondary_author>/<subject>
* <author>/<collection>/<media_type>/<secondary_author>/<subject>

## Output format

Produce a JSON array. Each element must include:

* `file_path` (string)
* `leaf` (string)
* `justification` (a short description for how a label fits in a classification)
* `classification` (one of the allowed labels)

## File output

Write the JSON array to a file in the `outputs` folder named **`classification.json`**, and update the `true_classification` field in the `classification` table

## Final task
Propose any revisions to the instructions to make them easier for an LLM to use
