# Data Model: Module 4 Vision-Language-Action

## Entity: ModuleChapter

### Attributes
- **id** (string): Unique identifier in kebab-case format for the module chapter
- **title** (string): Full title of the module chapter
- **module_number** (string): Module identifier (IV for Module 4)
- **chapter_number** (integer): Sequential number within the module
- **description** (string): Brief description of the chapter content
- **topics** (array): List of topics covered in the chapter
- **navigation_path** (string): Path for Docusaurus sidebar navigation
- **part_id** (string): Reference to Part V (Vision-Language-Action)
- **sections** (array): Required sections [overview, learning_outcomes, key_concepts, diagrams_code, labs_exercises]
- **created_date** (date): Date when the chapter was created
- **updated_date** (date): Date when the chapter was last updated

### Relationships
- Belongs to one Module (Module 4)
- Contains multiple ContentSections
- Related to other ModuleChapters as part of the sequential module structure

### Validation Rules
- id must be in kebab-case format
- module_number must be "IV" for Module 4
- chapter_number must be unique within the module
- topics array must not be empty
- all required sections must be present
- part_id must reference an existing Part V

## Entity: VLAToolRequirement

### Attributes
- **name** (string): Name of the VLA tool/component
- **version** (string): Recommended version of the tool
- **module_id** (string): Reference to Module 4
- **purpose** (string): Educational purpose of the tool in the module
- **installation_guide** (string): Reference to installation instructions
- **compatibility_matrix** (object): Compatible versions with other tools
- **license_type** (string): License type of the tool
- **created_date** (date): Date when the requirement was created
- **updated_date** (date): Date when the requirement was last updated

### Relationships
- Associated with one Module (Module 4)
- Referenced by multiple ModuleChapters that use the tool
- May have dependencies on other VLAToolRequirements

### Validation Rules
- name must be unique within the module
- version must follow semantic versioning
- module_id must reference Module 4
- purpose must not be empty

## Entity: ContentSection

### Attributes
- **id** (string): Unique identifier for the content section
- **chapter_id** (string): Reference to the parent ModuleChapter
- **section_type** (string): Type of section (overview, learning_outcomes, key_concepts, diagrams_code, labs_exercises)
- **title** (string): Title of the section
- **content** (string): Main content of the section
- **order** (integer): Display order within the chapter
- **required** (boolean): Whether this section is required
- **created_date** (date): Date when the section was created
- **updated_date** (date): Date when the section was last updated

### Relationships
- Belongs to one ModuleChapter
- Multiple ContentSections compose one ModuleChapter
- May reference external resources or examples

### Validation Rules
- section_type must be one of the allowed values
- order must be unique within the parent chapter
- content must not be empty if required is true
- chapter_id must reference an existing ModuleChapter

## Entity: VLAPipeline

### Attributes
- **id** (string): Unique identifier for the VLA pipeline
- **chapter_id** (string): Reference to the ModuleChapter containing this pipeline
- **title** (string): Title of the pipeline
- **description** (string): Brief description of the pipeline
- **components** (array): List of components in the pipeline (voice, language, action)
- **complexity_level** (string): Beginner, Intermediate, or Advanced
- **files** (array): List of file paths for the pipeline
- **instructions** (string): Step-by-step instructions
- **expected_outcome** (string): What the user should expect to see
- **created_date** (date): Date when the pipeline was created
- **updated_date** (date): Date when the pipeline was last updated

### Relationships
- Belongs to one ModuleChapter
- Multiple VLAPipelines may exist within one chapter
- May reference external resources or dependencies

### Validation Rules
- components array must contain at least one valid component
- complexity_level must be one of the allowed values
- files array must not be empty
- chapter_id must reference an existing ModuleChapter

## Entity: ModuleNavigation

### Attributes
- **id** (string): Unique identifier for the navigation item
- **module_id** (string): Reference to Module 4
- **label** (string): Display label for the navigation item
- **type** (string): Type of navigation item (category, doc, link)
- **items** (array): Child navigation items (for categories)
- **doc_id** (string): Reference to the document (for doc type)
- **href** (string): External link (for link type)
- **collapsible** (boolean): Whether the category is collapsible
- **collapsed** (boolean): Whether the category is initially collapsed
- **order** (integer): Display order in navigation

### Relationships
- Belongs to one Module (Module 4)
- Hierarchically organized to represent the module structure
- Links to ModuleChapter entities

### Validation Rules
- type must be one of category, doc, or link
- For category type, items array must not be empty
- For doc type, doc_id must reference an existing document
- For link type, href must be a valid URL
- module_id must reference Module 4

## Entity: CognitivePlan

### Attributes
- **id** (string): Unique identifier for the cognitive plan
- **module_id** (string): Reference to Module 4
- **name** (string): Name of the cognitive plan
- **description** (string): Brief description of the plan
- **input_type** (string): Type of input (natural language, command, etc.)
- **output_type** (string): Type of output (robot action, behavior sequence, etc.)
- **planning_algorithm** (string): Algorithm used for planning
- **components** (array): List of components involved in the plan
- **created_date** (date): Date when the plan was created
- **updated_date** (date): Date when the plan was last updated

### Relationships
- Associated with one Module (Module 4)
- May be referenced by multiple ModuleChapters
- Part of the cognitive planning framework for the module

### Validation Rules
- input_type must be a valid input type
- output_type must be a valid output type
- planning_algorithm must be a valid algorithm
- module_id must reference Module 4