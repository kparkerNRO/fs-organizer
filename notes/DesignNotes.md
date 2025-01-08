Folders view
    Used to get a wholistic view of a folder rename (not at a heirarchy level). A folder-first 
    way to check that a given folder is broken into the right categories
    (not supporting de-duplication of tags, or heirachy view)
    * Table of
        cleaned folder name
        classification
        original name
        categories
            maybe broken into subject/category/variant
            maybe just a list that caps out and inclues a [...] see more after 3(?) entries
    * Clicking opens a sub-table which displays
        ID
        cleaned name
        original name
        folder path
        categories
            should be able to click over to categories view

Category Grouping view
    Goal - validate grouping and renames, support breaking a category apart or merging categories
    * Table of
        category name (given a placeholder if we can't infer)
        classification
        counts
        matched subjects
        confidence score (for rename)
    * Clicking on a category expands to sub-table for eatch matching category
        * category name
        * original file name
        * classification
        * path
        * additional categories
        (might need a way to distinguish between, i.e. TC-wizard and CP-Wizard - a category "true" name as opposed to display name)
        (maybe visually group probably-related groupings?)

    possible actions:
        * rename category (overrides the name in all subcategories)
        * drag a sub-category to a new category (and auto rename)
        * create a category
        * extract a common name from a category
        * rename individual sub categories (or select a set and rename)
        * mark a grouping as invalid

    can filter by
        * classification
        * confidence score
        * category name
        * original file name (?)
    
    can sort by
        * original path
        * category name
        * original name
        * confidence score



Category view
    this represents the difference between "saved" classifications, and inferred ones
    * category name
    * classification
    * sub-classification
    * storage_state
        * saved (maybe a "locked/permanent" vs "this run"?)
        * inferred
    * for saved, something to indicate the confidence in the inference

    Actions:
        * sort and filter by classification, storage state, confidence
        * can mark an inferred class as a saved one (prevents future inference of that word)
        * can change the classification of a category

    Want to be able to save/commit and re-run after changes
    *maybe* want to keep a change log/events
        support undo/staged changes