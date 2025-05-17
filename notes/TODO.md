End state:
    From UI or CLI, supply directory. The app iterates through all files in the directory, inferring categories and groupings based on file path and name. Once the user confirms the re-org, moves all the files into a new heirarchy, with optional tagging features

    Front end that allows the user to review the categorizations done, and the proposed heirarchy

Features
    Backend
        [] Ability to load files from disk
        [] Categorize files
            [] based on file path
            [] based on file name
            [] based on type (?)

        Cateegorization feature
            [] Identify category "types" - grouping, variant, media type, etc
            [] De-duplicate and spelling correct categories
            [] generate folder heirarchies based on categories
            [] Flag possible matches for manual review/low confidence

    Frontend
        [] Provide path to load files on
        [] Provide interface to review categories
        [] Provide interface to review re-organization
        [] Issue the command to re-organize