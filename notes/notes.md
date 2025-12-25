Two types of confident
    How confident am I that these categories (or this category) applies to this folder
    How confident am I that this word is the cannonical name for the category

Stages and data structures
    1. gather:
        loads the folder structure into the db table
        represent folders & files
    2. classify
        breaks up folder names into path components, which are then classified by known terms and sub-categories
            folders -> have components (currently called folder-category)
            components have classifications
    3. group
        represents groups/categories which contain path-components
            the idea is to collapse path components into the same reference
        category = finalized/semi-finalized component names

        manual review step is to evaluate the grouping
            group
                path-component
                    path name
         
# Classification
Possible classifications for folder paths:
* primary author
* secondary author
* collection- representing a grouping of similar places or locations
* subject - representing a specific area eg: desert castle, or "fortune teller"
* media format - vtt, jpg, etc
* media type - audio, map, tokens, music, tiles, adventure
* variant - day, night, snow, sand, etc