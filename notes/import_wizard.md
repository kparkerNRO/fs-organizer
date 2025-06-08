Create new navigation item for an "import" view. This will be a "wizard" style import walkthrough to support human interaction in the ingestion pipeline.

You may use existing components, but create new pages, and create new mocked endpoints for all API calls that this uses.

The import should have the following steps:

1. (Import) Start the process with selecting the source folder. Once the folder is clicked, send the command to the backend to load, and display a loading icon while the backend processes the folder. When the process is complete, show a folder view with the imported data. User should press "next" to continue to the next stage

2. (Group) Display a loading icon until the backend call for grouped data returns (original data should be available immediately, but the grouped data will need time to process). Once this is ready, display the "grouped" heirarchy alongside the original folder structure. This page will allow interactive changes to the grouped structure, so it should have a "save" and a "next" button

3. (Organize) Once again, this will need time to load the data as the backend processes the group data. Once the grouped data is processed, show the original folder structure alongside the "organized" structure. Once again, this will have a "save" and a "next" button

4. (Review) No loading should be necessary for this one. Show the "finalized" folder structure, and present inputs for settigns:

- Target Folder
- Duplicate handling (Keep newest, Keep largest, Keep both, keep both if not identical)
  This is the last stage, and the button here is "apply". Once apply is clicked, display progress bar until complete.
