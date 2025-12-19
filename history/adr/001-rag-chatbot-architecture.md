# ADR 001: RAG Chatbot Architecture for Physical AI & Humanoid Robotics Book

## Status
Accepted

## Date
2025-12-18

## Context
We need to implement a Retrieval-Augmented Generation (RAG) chatbot for the Physical AI & Humanoid Robotics book that can answer questions about book content with proper citations. The system must be cost-effective, scalable, and production-ready while integrating cleanly with the existing Docusaurus documentation site.

## Decision
We will implement a multi-tier architecture using the following components:

1. **Frontend**: React chat component integrated with Docusaurus
2. **Backend**: FastAPI service for orchestration
3. **Vector Database**: Qdrant Cloud for content embeddings
4. **Metadata Store**: Neon Serverless Postgres for sessions and history
5. **Embeddings**: Qwen API for content vectorization
6. **LLM Service**: OpenRouter for response generation

## Alternatives Considered

### Alternative 1: OpenAI Stack
- Use OpenAI embeddings and GPT models
- Pros: Mature ecosystem, well-documented
- Cons: Higher cost, vendor lock-in, less experimentation with newer models

### Alternative 2: Self-hosted Solution
- Run all services locally (PGVector, local LLMs)
- Pros: Full control, no API costs
- Cons: Higher complexity, maintenance overhead, resource requirements

### Alternative 3: Single Provider (e.g., Pinecone + OpenAI)
- Use one vendor for all services
- Pros: Simplified integration
- Cons: Vendor lock-in, potentially higher costs

## Rationale
The chosen architecture provides the best balance of:
- Cost-effectiveness (Qwen embeddings and OpenRouter)
- Performance (Qdrant vector search optimized for similarity)
- Scalability (serverless components handle variable load)
- Flexibility (multiple LLM options through OpenRouter)
- Integration simplicity (standard APIs and protocols)

## Consequences

### Positive
- Cost-effective solution using competitive APIs
- Good performance with optimized vector database
- Flexible architecture allowing model experimentation
- Clean separation of concerns between components

### Negative
- Dependency on multiple external services
- Potential rate limiting from API providers
- Complexity of managing multiple service integrations

## Implementation Notes
- API keys and connection strings stored in environment variables
- Proper error handling for service outages
- Rate limiting and caching to manage costs
- Comprehensive monitoring and logging