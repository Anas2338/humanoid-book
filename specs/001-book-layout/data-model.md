# Data Model: Book Layout

## Entity: BookPart

### Attributes
- **id** (string): Unique identifier in kebab-case format for the book part
- **title** (string): Full title of the book part
- **part_number** (string): Roman numeral identifier (I, II, III, IV, V, VI, VII)
- **description** (string): Brief description of the part content
- **navigation_path** (string): Path for Docusaurus sidebar navigation
- **chapter_count** (integer): Number of chapters in this part (3-5)
- **created_date** (date): Date when the part was created
- **updated_date** (date): Date when the part was last updated

### Relationships
- Contains multiple ChapterStub entities
- Related to other BookPart entities as part of the sequential book structure

### Validation Rules
- id must be in kebab-case format
- part_number must be one of I, II, III, IV, V, VI, VII
- title must be unique across all BookPart entities
- chapter_count must be between 3 and 5 inclusive

## Entity: ChapterStub

### Attributes
- **id** (string): Unique identifier in kebab-case format
- **title** (string): Title of the chapter
- **part_id** (string): Reference to the parent BookPart
- **chapter_number** (integer): Sequential number within the part
- **overview** (string): Chapter overview content
- **learning_outcomes** (array): List of learning outcomes for the chapter
- **key_concepts** (array): List of key concepts covered in the chapter
- **diagrams_code** (string): Placeholder for diagrams and code sections
- **labs_exercises** (array): List of labs and exercises for the chapter
- **status** (string): Current status (stub, in_progress, completed)
- **created_date** (date): Date when the stub was created
- **updated_date** (date): Date when the stub was last updated

### Relationships
- Belongs to one BookPart
- Multiple ChapterStubs compose one BookPart

### Validation Rules
- id must be in kebab-case format
- chapter_number must be unique within the parent BookPart
- All required sections (overview, learning_outcomes, key_concepts, diagrams_code, labs_exercises) must be present
- part_id must reference an existing BookPart

## Entity: DocusaurusNavigation

### Attributes
- **id** (string): Unique identifier for the navigation item
- **label** (string): Display label for the navigation item
- **type** (string): Type of navigation item (category, doc, link)
- **items** (array): Child navigation items (for categories)
- **doc_id** (string): Reference to the document (for doc type)
- **href** (string): External link (for link type)
- **collapsible** (boolean): Whether the category is collapsible
- **collapsed** (boolean): Whether the category is initially collapsed

### Relationships
- Hierarchically organized to represent the book structure
- Links to ChapterStub and BookPart entities

### Validation Rules
- type must be one of category, doc, or link
- For category type, items array must not be empty
- For doc type, doc_id must reference an existing document
- For link type, href must be a valid URL

## Entity: ContentTemplate

### Attributes
- **id** (string): Unique identifier for the template
- **name** (string): Name of the template
- **type** (string): Type of content the template applies to (chapter, part, appendix)
- **content** (string): Template content with placeholders
- **required_sections** (array): List of required sections in the template
- **metadata_fields** (array): List of metadata fields required
- **created_date** (date): Date when the template was created
- **updated_date** (date): Date when the template was last updated

### Relationships
- Used to generate ChapterStub entities
- Applied during the creation of book parts

### Validation Rules
- type must be one of chapter, part, or appendix
- required_sections must not be empty for chapter templates
- All placeholders in content must be properly formatted

## Entity: DeploymentConfiguration

### Attributes
- **id** (string): Unique identifier for the deployment config
- **platform** (string): Deployment platform (GitHub Pages, Netlify, etc.)
- **branch** (string): Branch to deploy from
- **build_command** (string): Command to build the site
- **output_directory** (string): Directory containing built files
- **domain** (string): Custom domain if applicable
- **ssl_enabled** (boolean): Whether SSL is enabled
- **created_date** (date): Date when the config was created
- **updated_date** (date): Date when the config was last updated

### Relationships
- Associated with the overall book project
- References the Docusaurus site configuration

### Validation Rules
- platform must be a supported deployment platform
- build_command must be a valid command
- output_directory must exist in the project