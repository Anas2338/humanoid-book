# Data Model: Appendices Module

## Entity: AppendixChapter

### Attributes
- **id** (string): Unique identifier in kebab-case format for the appendix chapter
- **title** (string): Full title of the appendix chapter
- **appendix_number** (string): Sequential appendix identifier (A, B, C, D, E)
- **description** (string): Brief description of the appendix content
- **navigation_path** (string): Path for Docusaurus sidebar navigation
- **sections** (array): List of required sections in the appendix
- **created_date** (date): Date when the appendix was created
- **updated_date** (date): Date when the appendix was last updated
- **version** (string): Version of the appendix content

### Relationships
- Related to the main book through navigation structure
- Connected to other appendices as part of the appendices module

### Validation Rules
- id must be in kebab-case format
- appendix_number must be a single uppercase letter (A-E)
- title must be unique within the appendices module
- sections must contain all required sections as defined in the specification

## Entity: AppendixContentSection

### Attributes
- **section_type** (string): Type of section (overview, tools_resources, diagrams_stubs, reference_tables, etc.)
- **content** (string): The actual content of the section
- **appendix_id** (string): Reference to the parent AppendixChapter
- **order** (integer): Order of the section within the appendix
- **required** (boolean): Whether the section is required

### Relationships
- Belongs to one AppendixChapter
- Multiple sections compose one AppendixChapter

### Validation Rules
- section_type must be one of the predefined types
- order must be unique within the parent AppendixChapter
- required sections must be present in all AppendixChapter instances

## Entity: ToolVersionRequirement

### Attributes
- **tool_name** (string): Name of the tool (ROS 2, Gazebo, Unity, Isaac)
- **minimum_version** (string): Minimum required version
- **recommended_version** (string): Recommended version for compatibility
- **appendix_id** (string): Reference to the AppendixChapter that documents this tool
- **status** (string): Status of the version requirement (active, deprecated, experimental)

### Relationships
- Associated with Appendix A (Development Environment Setup)
- May be referenced by other appendices that use these tools

### Validation Rules
- minimum_version must be a valid semantic version
- recommended_version must be greater than or equal to minimum_version
- tool_name must be one of the specified tools

## Entity: TroubleshootingEntry

### Attributes
- **issue_title** (string): Brief title of the troubleshooting issue
- **description** (string): Detailed description of the issue
- **diagnostic_steps** (array): Steps to diagnose the issue
- **resolution_steps** (array): Steps to resolve the issue
- **tool_affected** (string): Which tool the issue relates to
- **severity** (string): Severity level (low, medium, high, critical)
- **appendix_id** (string): Reference to Appendix C (Troubleshooting Guide)

### Relationships
- Belongs to Appendix C (Troubleshooting Guide)
- Related to specific tools in the tool ecosystem

### Validation Rules
- severity must be one of the predefined values
- diagnostic_steps and resolution_steps must not be empty
- tool_affected must be a valid tool name

## Entity: ReferenceEntry

### Attributes
- **title** (string): Title of the reference
- **authors** (string): Author names
- **publication_date** (date): Date of publication
- **source_type** (string): Type of source (academic, technical, blog, etc.)
- **url** (string): URL to the reference (if available)
- **citation_format** (string): IEEE formatted citation
- **appendix_id** (string): Reference to Appendix E (References & Further Reading)
- **peer_reviewed** (boolean): Whether the source is peer-reviewed

### Relationships
- Belongs to Appendix E (References & Further Reading)
- May be referenced by other appendices that cite this source

### Validation Rules
- citation_format must follow IEEE standards
- peer_reviewed must be boolean
- At least 40% of entries in the appendix must have peer_reviewed = true