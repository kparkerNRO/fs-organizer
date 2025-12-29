End state:
    From UI or CLI, supply directory. The app iterates through all files in the directory, inferring categories and groupings based on file path and name. Once the user confirms the re-org, moves all the files into a new heirarchy, with optional tagging features

    Front end that allows the user to review the categorizations done, and the proposed heirarchy

Features
    Backend
        [x] Ability to load files from disk
        [] Categorize files
            [] based on file path
            [] based on file name
            [] based on type (?)

        Cateegorization feature
            [x] Identify category "types" - grouping, variant, media type, etc
            [x] De-duplicate and spelling correct categories
            [x] generate folder heirarchies based on categories
            [x] Flag possible matches for manual review/low confidence

    Frontend
        [x] Provide path to load files on
        [x] Provide interface to review categories
        [] Provide interface to review re-organization
        [] Issue the command to re-organize


--------------
# Next steps
* Frontend
    * Add context menus to add and remove folders
    * validate that moving works correctly when folders should merge
    * Filter visualization down to only non-confident answers
        * Add some indication in a folder tree that there are non-confident children
    * Disable editing on non-editable windows (except delete/remove)
    * Add toast messages for operation failure and delete warning
    * Implement save & undo
    * Allow shift-ctl multi-select
    

* Backend
    * implement save
    * figure out why ebooks aren't loading into database

* General
    * Investigate allowing the user to load multiple folders at the start