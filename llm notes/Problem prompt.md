We have a large directory files from different Patreon creators, and we want to organize them consistently. The files they generally come in several variants (day, night, snow, etc) as well as formats (gridded vs non-gridded, optimized for VTT or for print, or different file formats). Additionally some sets of maps may be grouped into hierarchical collections. All of this information is encoded in the supplied file name or folder hierarchy. 

Some examples:
* for one creator, the files will be organized into the nested collections "Into the wilds" -> "town" and then the maps will be organized under "house" -> VTT -> gridded -> winter/summer/night/etc
* For another creator, the files will be organized "Dungeons" -> "VTT" -> "dragon lair" -> day/night/etc, and the individual files will be "gridded" or "no-grid"

Some known features of the data:
* No creator organizes their collections more than three deep ("into the wilds"/"town", "Dungeons"/"dragon lair", "town of fairview"/"ruined"/"upper district", etc)
* We generally know the names of the variants, but the subject and the collections must be inferred from the file path. However, variants may be combined into a single folder (eg: "winter night"), and we want to split them out ("winter"/"night")
* Some information is encoded redundantly: a file might be stored under "Dungons/dragon lairs/green dragon lair/entrance/night", but "dragon lair" is specified redundantly, we want to simplify that set of folders to "dragon lair/green dragon"
* Some files may be near-matches which need to be flagged as belonging to the same collection: "The wizard tower" and "Wizards tower" should be merged into a cannonical name such as "The Wizard tower"
* However, the previous two rules interact - we may have a folder "Wizard tower" and another "Wizard Tower Interior" -> these actually represent a herarchical relationship of "Wizard tower" and "Wizard Tower/Interior"



Our goal is to **reorganize** these files into a consistent hierarchy according to the following logic:

1. **Classification**  
   - We want to label each **folder** as one of:
     - **Variant**: If its name consists entirely of known variant tokens (e.g., “day,” “night,” “winter,” “clean”).  
     - **Collection**: If the folder name occurs frequently at the same or higher level (implying a broad grouping like “Dungeons” or “Towns”).  
     - **Subject**: If it is a unique name and all its children are variants (e.g., “green dragon lair” contains only “winter day,” “night,” etc.).  
     - **Uncertain**: If it doesn’t clearly fit the above.  

2. **Synonym Grouping**  
   - We need to group near-duplicate folder names under a single “suspected group” so we can unify “wizard tower,” “the wizard’s tower,” and “wizards tower” as the same concept.  
   - However, we must avoid merging cases like “wizard tower interior” (a sub-location) into “wizard tower,” which should remain distinct.

3. **Re-Organization**  
   - After classification and grouping, we want to generate (or apply) a **new folder hierarchy**. For instance:
     1. Move or rename collections to the top level.  
     2. Nest each subject under its collection.  
     3. Nest variants beneath each subject.  
   - Any remaining “uncertain” items should go to a “to-review” or “misc” location for manual inspection

We are looking for a plan or algorithmic approach that:
- Processes a large, messy directory of files and folders.  
- Stores the metadata in a database for easy reference and minimal repeated disk I/O.  
- Classifies folders as variant, collection, subject, or uncertain based on structural and frequency-based heuristics.  
- Groups near-duplicates without incorrectly merging sub-locations.  
- Produces a final re-organized structure and either renames/moves the actual files or provides a mapping for a subsequent “commit” step.

We want a language-indepentent step-by-step solution or strategy, from ingestion (reading the raw file system) through classification and grouping (in the database), to ultimately **reorganizing** or **renaming** the files/folders according to this consistent hierarchy.  

Break this problem down into parts, and think through the solution step by step, using concrete examples.