# Research Findings: RAG Chatbot for Physical AI & Humanoid Robotics Book

## Decision 1: Qwen Embeddings Model Selection

**Decision**: Use Qwen2-7B-Instruct embeddings for the RAG system
**Rationale**: Qwen2-7B-Instruct provides a good balance of quality and cost-effectiveness. It's specifically designed for understanding and generating human-like text, making it suitable for educational content. The model is available through OpenRouter's API which we're already using for LLM services.

**Alternatives considered**:
- Sentence Transformers (all-MiniLM-L6-v2): Lower cost but potentially less accurate for technical content
- OpenAI embeddings (text-embedding-3-small): Higher cost and vendor lock-in
- Custom embeddings model: Higher development time and maintenance

## Decision 2: Book Content Structure and Chunking Strategy

**Decision**: Use a hierarchical chunking approach with semantic boundaries
**Rationale**: The book content should be chunked at semantic boundaries (sections, subsections) rather than fixed token counts to maintain context and improve citation accuracy. This approach ensures that retrieved content is meaningful and properly attributable.

**Chunking Strategy**:
- Primary chunks: Book sections (h2 headings)
- Secondary chunks: Subsections (h3 headings) if primary chunks are too large (>500 tokens)
- Metadata: Include chapter, section, page number, and source file in each chunk
- Overlap: 10% overlap between chunks to maintain context across boundaries

## Decision 3: Docusaurus Integration Approach

**Decision**: Integrate OpenAI ChatKit with Docusaurus
**Rationale**: OpenAI ChatKit provides a production-ready chat interface with built-in features like typing indicators, message history, and responsive design. This reduces development time and ensures a high-quality user experience.

**Implementation approach**:
- Install and configure OpenAI ChatKit in the Docusaurus project
- Customize ChatKit appearance to match Docusaurus theme
- Implement custom functionality for text selection and citation display
- Use Docusaurus' plugin system to inject ChatKit component
- Ensure mobile-responsive design

## Decision 5: OpenAI ChatKit Customization

**Decision**: Customize OpenAI ChatKit for educational RAG use case
**Rationale**: While ChatKit provides a solid foundation, it needs customization to properly display citations to book content and handle the selected text functionality required by the spec.

**Customization approach**:
- Override default message rendering to include citation links
- Add custom input handling for selected text context
- Implement custom styling to match Docusaurus color scheme
- Add educational-specific UI elements like source attribution

## Decision 4: API Cost and Rate Limit Analysis

**Decision**: Implement usage monitoring and caching to manage costs
**Rationale**: Based on OpenRouter pricing and Qwen API availability, costs are manageable for expected usage but require monitoring to prevent unexpected expenses.

**Cost Analysis**:
- Qwen embeddings: Approximately $0.008/1K tokens (estimated)
- OpenRouter LLM calls: Varies by model, approximately $0.005-0.05 per 1K tokens
- Qdrant Cloud: Free tier supports up to 5 vectors for testing
- Neon Postgres: Free tier available

**Rate Limits**:
- OpenRouter: Varies by model, typically 100-1000 RPM
- Qwen: Expected to be sufficient for educational use case
- Implementation will include retry logic and rate limiting

## Decision 5: Session Management Strategy

**Decision**: Use anonymous sessions with optional account persistence
**Rationale**: For educational use, anonymous sessions provide immediate access while optional accounts allow users to maintain conversation history across devices.

**Implementation**:
- Anonymous sessions: Identified by browser session/local storage
- Optional accounts: For users who want to save conversations across devices
- Session timeout: 24 hours of inactivity before cleanup

## Decision 6: Vector Dimension and Configuration

**Decision**: Use 1536-dimensional embeddings from Qwen model
**Rationale**: Qwen embeddings typically produce 1536-dimensional vectors which provide good quality while being efficient to store and search. This dimensionality is standard and supported by Qdrant.

**Configuration**:
- Vector size: 1536 dimensions
- Distance metric: Cosine similarity
- Index type: HNSW for efficient similarity search
- Batch size: 10-50 for initial indexing