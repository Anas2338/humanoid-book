# Data Model: Module 3 AI-Robot Brain

## Entity: ModuleChapter

### Attributes
- **id** (string): Unique identifier in kebab-case format for the module chapter
- **title** (string): Full title of the module chapter
- **module_number** (string): Module identifier (III for Module 3)
- **chapter_number** (integer): Sequential number within the module
- **description** (string): Brief description of the chapter content
- **topics** (array): List of topics covered in the chapter
- **navigation_path** (string): Path for Docusaurus sidebar navigation
- **part_id** (string): Reference to Part IV (AI-Robot Brain)
- **sections** (array): Required sections [overview, learning_outcomes, key_concepts, diagrams_code, labs_exercises]
- **created_date** (date): Date when the chapter was created
- **updated_date** (date): Date when the chapter was last updated

### Relationships
- Belongs to one Module (Module 3)
- Contains multiple ContentSections
- Related to other ModuleChapters as part of the sequential module structure

### Validation Rules
- id must be in kebab-case format
- module_number must be "III" for Module 3
- chapter_number must be unique within the module
- topics array must not be empty
- all required sections must be present
- part_id must reference an existing Part IV

## Entity: IsaacToolRequirement

### Attributes
- **name** (string): Name of the Isaac tool/component
- **version** (string): Recommended version of the tool
- **module_id** (string): Reference to Module 3
- **purpose** (string): Educational purpose of the tool in the module
- **installation_guide** (string): Reference to installation instructions
- **compatibility_matrix** (object): Compatible versions with other tools
- **license_type** (string): License type of the tool
- **created_date** (date): Date when the requirement was created
- **updated_date** (date): Date when the requirement was last updated

### Relationships
- Associated with one Module (Module 3)
- Referenced by multiple ModuleChapters that use the tool
- May have dependencies on other IsaacToolRequirements

### Validation Rules
- name must be unique within the module
- version must follow semantic versioning
- module_id must reference Module 3
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

## Entity: IsaacExample

### Attributes
- **id** (string): Unique identifier for the Isaac example
- **chapter_id** (string): Reference to the ModuleChapter containing this example
- **title** (string): Title of the example
- **description** (string): Brief description of the example
- **isaac_tool** (string): Primary tool used (Isaac Sim, Isaac ROS, Nav2, etc.)
- **complexity_level** (string): Beginner, Intermediate, or Advanced
- **files** (array): List of file paths for the example
- **instructions** (string): Step-by-step instructions
- **expected_outcome** (string): What the user should expect to see
- **created_date** (date): Date when the example was created
- **updated_date** (date): Date when the example was last updated

### Relationships
- Belongs to one ModuleChapter
- Multiple IsaacExamples may exist within one chapter
- May reference external resources or dependencies

### Validation Rules
- isaac_tool must be a valid Isaac component
- complexity_level must be one of the allowed values
- files array must not be empty
- chapter_id must reference an existing ModuleChapter

## Entity: ModuleNavigation

### Attributes
- **id** (string): Unique identifier for the navigation item
- **module_id** (string): Reference to Module 3
- **label** (string): Display label for the navigation item
- **type** (string): Type of navigation item (category, doc, link)
- **items** (array): Child navigation items (for categories)
- **doc_id** (string): Reference to the document (for doc type)
- **href** (string): External link (for link type)
- **collapsible** (boolean): Whether the category is collapsible
- **collapsed** (boolean): Whether the category is initially collapsed
- **order** (integer): Display order in navigation

### Relationships
- Belongs to one Module (Module 3)
- Hierarchically organized to represent the module structure
- Links to ModuleChapter entities

### Validation Rules
- type must be one of category, doc, or link
- For category type, items array must not be empty
- For doc type, doc_id must reference an existing document
- For link type, href must be a valid URL
- module_id must reference Module 3

## Entity: GPURequirement

### Attributes
- **id** (string): Unique identifier for the GPU requirement
- **module_id** (string): Reference to Module 3
- **gpu_type** (string): Type of GPU required (NVIDIA, etc.)
- **compute_capability** (string): CUDA compute capability version
- **memory_requirement** (integer): Minimum GPU memory in GB
- **driver_version** (string): Required driver version
- **cuda_version** (string): Required CUDA version
- **purpose** (string): Educational purpose of the GPU requirement
- **created_date** (date): Date when the requirement was created
- **updated_date** (date): Date when the requirement was last updated

### Relationships
- Associated with one Module (Module 3)
- May be referenced by multiple IsaacToolRequirements
- Part of the hardware specification for the module

### Validation Rules
- gpu_type must be a supported GPU type
- compute_capability must follow CUDA versioning
- memory_requirement must be positive
- module_id must reference Module 3