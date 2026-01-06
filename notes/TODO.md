End state:
    From UI or CLI, supply directory. The app iterates through all files in the directory, inferring categories and groupings based on file path and name. Once the user confirms the re-org, moves all the files into a new heirarchy, with optional tagging features

    Front end that allows the user to review the categorizations done, and the proposed heirarchy


# Next steps
* Feature: Folder structure
    * Rework API to use the node/hierarchy split - send nodes and multiple hierarchys. Think github diff traking for the structures
    * Implement sync between the two views

* Feature: manual review
    * frontend
        * Add context menus to add folders
        * validate that moving works correctly when folders should merge
        * Filter visualization down to only non-confident answers
            * Add some indication in a folder tree that there are non-confident children
        * Disable editing on non-editable windows (except delete/remove)
        * Add toast messages for operation failure and delete warning
        * Implement save & undo
        * Add some hints about the ui - help future-self remember how to use the various features
        * Allow manual folder editing
    * Backend
        * implement save/commit after manual review


* Feature: categorize
    * rework grouping to a very basic "common token" grouping - rather than a full NLP grouping, do common token string match 
        * explore normalizing before grouping
        * confidence is probably the ratio of First part: second part (second part should be < 50%)
    * Add some additional heuristic categorizing to filter out things we know for sure
    * Train model on the remaining data
        * I suspect "term count" is a useful feature for the model to use
        * I also suspect we'll want to add files into the mix, but that is a new feature entirely
    * rework frontend "categorize" review to be category centric rather than folder centric - maybe show original folder to category mapping for review

* General
    * Investigate allowing the user to load multiple folders at the start
    * Pass around lists rather than database sessions (databases are slow and lots of sessions)
    * Clean out all of the " | none " types which we can promise values for

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
        
        ML feature:
            [x] Infrastructure for training model
            [x] Ability to run model against training set
            [] Integrate training set with output of grouping


    Frontend
        [x] Provide path to load files on
        [x] Provide interface to review categories
        [] Provide interface to review re-organization
        [] Issue the command to re-organize


--------------
