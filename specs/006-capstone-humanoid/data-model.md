# Data Model: Capstone Autonomous Humanoid

## Entity: ModuleChapter

### Attributes
- **id** (string): Unique identifier in kebab-case format for the module chapter
- **title** (string): Full title of the module chapter
- **module_number** (string): Module identifier (VI for Capstone)
- **chapter_number** (integer): Sequential number within the module
- **description** (string): Brief description of the chapter content
- **topics** (array): List of topics covered in the chapter
- **navigation_path** (string): Path for Docusaurus sidebar navigation
- **part_id** (string): Reference to Part VI (Capstone)
- **sections** (array): Required sections [overview, learning_outcomes, key_concepts, diagrams_code, labs_exercises]
- **created_date** (date): Date when the chapter was created
- **updated_date** (date): Date when the chapter was last updated

### Relationships
- Belongs to one Module (Capstone)
- Contains multiple ContentSections
- Related to other ModuleChapters as part of the sequential module structure

### Validation Rules
- id must be in kebab-case format
- module_number must be "VI" for Capstone
- chapter_number must be unique within the module
- topics array must not be empty
- all required sections must be present
- part_id must reference an existing Part VI

## Entity: CapstoneToolRequirement

### Attributes
- **name** (string): Name of the capstone tool/component
- **version** (string): Recommended version of the tool
- **module_id** (string): Reference to Capstone module
- **purpose** (string): Educational purpose of the tool in the module
- **installation_guide** (string): Reference to installation instructions
- **compatibility_matrix** (object): Compatible versions with other tools
- **license_type** (string): License type of the tool
- **created_date** (date): Date when the requirement was created
- **updated_date** (date): Date when the requirement was last updated

### Relationships
- Associated with one Module (Capstone)
- Referenced by multiple ModuleChapters that use the tool
- May have dependencies on other CapstoneToolRequirements

### Validation Rules
- name must be unique within the module
- version must follow semantic versioning
- module_id must reference Capstone module
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

## Entity: IntegrationComponent

### Attributes
- **id** (string): Unique identifier for the integration component
- **chapter_id** (string): Reference to the ModuleChapter containing this component
- **name** (string): Name of the integration component
- **description** (string): Brief description of the component
- **module_origin** (string): Originating module (ROS 2, Gazebo, Isaac, VLA)
- **complexity_level** (string): Beginner, Intermediate, or Advanced
- **files** (array): List of file paths for the component
- **instructions** (string): Step-by-step instructions
- **expected_outcome** (string): What the user should expect to see
- **created_date** (date): Date when the component was created
- **updated_date** (date): Date when the component was last updated

### Relationships
- Belongs to one ModuleChapter
- Multiple IntegrationComponents may exist within one chapter
- May reference external resources or dependencies

### Validation Rules
- module_origin must be a valid module identifier
- complexity_level must be one of the allowed values
- files array must not be empty
- chapter_id must reference an existing ModuleChapter

## Entity: ModuleNavigation

### Attributes
- **id** (string): Unique identifier for the navigation item
- **module_id** (string): Reference to Capstone module
- **label** (string): Display label for the navigation item
- **type** (string): Type of navigation item (category, doc, link)
- **items** (array): Child navigation items (for categories)
- **doc_id** (string): Reference to the document (for doc type)
- **href** (string): External link (for link type)
- **collapsible** (boolean): Whether the category is collapsible
- **collapsed** (boolean): Whether the category is initially collapsed
- **order** (integer): Display order in navigation

### Relationships
- Belongs to one Module (Capstone)
- Hierarchically organized to represent the module structure
- Links to ModuleChapter entities

### Validation Rules
- type must be one of category, doc, or link
- For category type, items array must not be empty
- For doc type, doc_id must reference an existing document
- For link type, href must be a valid URL
- module_id must reference Capstone module

## Entity: CapstoneProject

### Attributes
- **id** (string): Unique identifier for the capstone project
- **module_id** (string): Reference to Capstone module
- **name** (string): Name of the capstone project
- **description** (string): Brief description of the project
- **objectives** (array): List of project objectives
- **requirements** (array): List of project requirements
- **evaluation_criteria** (array): List of evaluation criteria
- **timeline** (object): Project timeline with milestones
- **created_date** (date): Date when the project was created
- **updated_date** (date): Date when the project was last updated

### Relationships
- Associated with one Module (Capstone)
- May be referenced by multiple ModuleChapters
- Part of the capstone framework for the module

### Validation Rules
- objectives array must not be empty
- requirements array must not be empty
- evaluation_criteria array must not be empty
- module_id must reference Capstone module