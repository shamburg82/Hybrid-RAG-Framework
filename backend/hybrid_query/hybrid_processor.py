# hybrid_query/hybrid_processor.py
"""
Hybrid Query Processor - Core implementation for the enhanced RAG system

This processor determines query type and routes to appropriate data sources:
- Structured queries: For exhaustive extractions (all analyses, all criteria)
- Semantic queries: For conceptual/guidance questions  
- Graph queries: For relationship and cross-reference questions
- Hybrid queries: Combining multiple approaches for comprehensive answers
"""

import re
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass
import datetime

from llama_index.core import Document
from tiered_rag_system import TieredRAGSystem, StorageTier
from structured_extractors.base_extractor import StructuredData
from knowledge_graph.graph_query_engine import GraphQueryEngine
from validation.confidence_scorer import ConfidenceScorer


class QueryType(Enum):
    """Types of queries supported by the hybrid system."""
    EXHAUSTIVE = "exhaustive"      # Complete extractions from structured data
    SEMANTIC = "semantic"          # Traditional RAG for conceptual queries
    GRAPH = "graph"               # Knowledge graph for relationships
    HYBRID = "hybrid"             # Combined approach
    CROSS_REFERENCE = "cross_reference"  # Validation and compliance


@dataclass
class QueryIntent:
    """Represents the detected intent of a query."""
    query_type: QueryType
    confidence: float
    extraction_target: Optional[str] = None  # What to extract (e.g., "inclusion_criteria")
    domain_focus: Optional[str] = None       # Domain focus (e.g., "protocol", "sap")
    scope: str = "specific"                  # "specific", "comprehensive", "exhaustive"
    requires_validation: bool = False


@dataclass
class HybridResult:
    """Result from hybrid query processing."""
    query: str
    query_type: QueryType
    primary_response: str
    structured_data: Optional[Dict[str, Any]] = None
    semantic_results: Optional[Dict[str, Any]] = None
    graph_results: Optional[Dict[str, Any]] = None
    confidence_scores: Dict[str, float] = None
    completeness_score: float = 0.0
    sources: List[Dict[str, Any]] = None
    validation_results: Optional[Dict[str, Any]] = None
    processing_time: float = 0.0
    timestamp: str = None

class QueryClassifier:
    """Classifies queries to determine optimal processing strategy."""
    
    def __init__(self):
        self.exhaustive_patterns = [
            r'\ball\s+(?:analyses|criteria|endpoints|specifications|variables|domains)',
            r'(?:complete|full|entire)\s+(?:list|set)\s+of',
            r'(?:extract|retrieve|list)\s+(?:all|every)',
            r'enumerate\s+(?:all|every)',
            r'(?:what\s+are\s+all|give\s+me\s+all)',
            r'comprehensive\s+(?:list|summary)\s+of'
        ]
        
        self.semantic_patterns = [
            r'(?:how\s+(?:should|do|can)|what\s+is|explain|describe)',
            r'(?:guidance|recommendation|best\s+practice)',
            r'(?:principle|concept|approach|methodology)',
            r'(?:why|when|where)\s+(?:should|would|can)'
        ]
        
        self.graph_patterns = [
            r'(?:relationship|connection|link)\s+between',
            r'(?:related|associated|connected)\s+to',
            r'(?:cross.?reference|validation|compliance)',
            r'(?:depends?\s+on|requires|based\s+on)',
            r'(?:hierarchy|structure|organization)'
        ]
        
        self.extraction_targets = {
            'protocol_summary': ['synopsis', 'title', 'brief title', 'rationale'],
            'estimands': ['estimands', 'intercurrent events'],
            'study_design': ['study design', 'overall design'],
            'schedule_of_assessments': ['schedule of assessments', 'schedule of visits', 'schedule of activities', 'visit schedule'],
            'planned_procedures': ['study assessments', 'study procedures'],
            'inclusion_criteria': ['inclusion', 'eligible', 'entry criteria'],
            'exclusion_criteria': ['exclusion', 'excluded', 'ineligible'],
            'study_treatment': ['study interventions', 'dosing', 'study drug', 'study therapy', 'study treatment'],
            'discontinuation': ['study discontinuation', 'treatment discontinuation', 'withdrawal', 'lost to follow-up'],
            'adverse_events': ['adverse events', 'adverse reactions', 'reportable events', 'reportable experiences'],
            'pk_pg': ['pharmacokinetics', 'pharmacogenomics'],
            'endpoints': ['endpoint', 'outcome', 'efficacy measure', 'safety measure'],
            'analyses': ['statistical considerations', 'analysis', 'statistical method', 'analytic approach']

        }
    
    def classify_query(self, query: str) -> QueryIntent:
        """Classify query and determine processing strategy."""
        query_lower = query.lower()
        
        # Detect query type
        exhaustive_score = self._pattern_score(query_lower, self.exhaustive_patterns)
        semantic_score = self._pattern_score(query_lower, self.semantic_patterns)
        graph_score = self._pattern_score(query_lower, self.graph_patterns)
        
        # Determine primary type
        scores = {
            QueryType.EXHAUSTIVE: exhaustive_score,
            QueryType.SEMANTIC: semantic_score,
            QueryType.GRAPH: graph_score
        }
        
        primary_type = max(scores, key=scores.get)
        max_score = scores[primary_type]
        
        # If scores are close, use hybrid approach
        if self._scores_are_close(scores.values()):
            primary_type = QueryType.HYBRID
            max_score = sum(scores.values()) / len(scores)
        
        # Detect extraction target
        extraction_target = self._detect_extraction_target(query_lower)
        
        # Detect domain focus
        domain_focus = self._detect_domain_focus(query_lower)
        
        # Determine scope
        scope = self._determine_scope(query_lower)
        
        # Check if validation required
        requires_validation = any(word in query_lower for word in [
            'validate', 'verify', 'check', 'confirm', 'compliance', 'consistent'
        ])
        
        return QueryIntent(
            query_type=primary_type,
            confidence=max_score,
            extraction_target=extraction_target,
            domain_focus=domain_focus,
            scope=scope,
            requires_validation=requires_validation
        )
    
    def _pattern_score(self, text: str, patterns: List[str]) -> float:
        """Calculate pattern matching score."""
        matches = 0
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                matches += 1
        return matches / len(patterns) if patterns else 0.0
    
    def _scores_are_close(self, scores: List[float], threshold: float = 0.3) -> bool:
        """Check if scores are close enough to warrant hybrid approach."""
        max_score = max(scores)
        return sum(1 for score in scores if abs(score - max_score) <= threshold) >= 2
    
    def _detect_extraction_target(self, query: str) -> Optional[str]:
        """Detect what should be extracted from the query."""
        for target, keywords in self.extraction_targets.items():
            if any(keyword in query for keyword in keywords):
                return target
        return None
    
    def _detect_domain_focus(self, query: str) -> Optional[str]:
        """Detect domain focus (protocol, SAP, etc.)."""
        domain_keywords = {
            'protocol': ['protocol', 'study design', 'inclusion', 'exclusion', 'treatment'],
            'sap': ['sap', 'statistical', 'analysis plan', 'population', 'method'],
            'define_xml': ['define.xml', 'metadata', 'domain', 'variable', 'dataset'],
            'cdisc': ['cdisc', 'sdtm', 'adam', 'standard', 'implementation guide']
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in query for keyword in keywords):
                return domain
        return None
    
    def _determine_scope(self, query: str) -> str:
        """Determine query scope."""
        if any(word in query for word in ['all', 'every', 'complete', 'comprehensive', 'entire']):
            return 'exhaustive'
        elif any(word in query for word in ['summary', 'overview', 'key', 'main', 'primary']):
            return 'comprehensive'
        else:
            return 'specific'


class HybridQueryProcessor:
    """Main hybrid query processor coordinating different query strategies."""
    
    def __init__(self, rag_system: TieredRAGSystem, structured_store, 
                 graph_engine: GraphQueryEngine = None):
        self.rag_system = rag_system
        self.structured_store = structured_store
        self.graph_engine = graph_engine
        self.query_classifier = QueryClassifier()
        self.confidence_scorer = ConfidenceScorer()
    
    def process_query(self, query: str, study_id: Optional[str] = None,
                     query_type: Optional[QueryType] = None,
                     top_k: int = 5) -> HybridResult:
        """Process query using hybrid approach."""
        start_time = datetime.datetime.now()
        
        # Classify query if type not provided
        if query_type is None:
            query_intent = self.query_classifier.classify_query(query)
            query_type = query_intent.query_type
        else:
            query_intent = QueryIntent(query_type=query_type, confidence=1.0)
        
        print(f"🔍 Processing {query_type.value} query: {query}")
        if query_intent.extraction_target:
            print(f"   Target: {query_intent.extraction_target}")
        if query_intent.domain_focus:
            print(f"   Domain: {query_intent.domain_focus}")
        
        # Route to appropriate processing method
        if query_type == QueryType.EXHAUSTIVE:
            result = self._query_structured_data(query, query_intent, study_id)
        elif query_type == QueryType.SEMANTIC:
            result = self._query_vector_store(query, query_intent, study_id, top_k)
        elif query_type == QueryType.GRAPH:
            result = self._query_knowledge_graph(query, query_intent, study_id)
        elif query_type == QueryType.HYBRID:
            result = self._process_hybrid_query(query, query_intent, study_id, top_k)
        else:
            result = self._query_vector_store(query, query_intent, study_id, top_k)
        
        # Calculate processing time
        processing_time = (datetime.datetime.now() - start_time).total_seconds()
        result.processing_time = processing_time
        result.timestamp = datetime.datetime.now().isoformat()
        
        # Add validation if required
        if query_intent.requires_validation:
            result.validation_results = self._validate_results(result, study_id)
        
        return result
    
    def _query_structured_data(self, query: str, intent: QueryIntent, 
                             study_id: Optional[str]) -> HybridResult:
        """Query structured data for exhaustive extractions."""
        print("   📊 Querying structured data store...")
        
        structured_results = {}
        confidence_scores = {}
        all_sources = []
        
        # Query structured store based on extraction target
        if intent.extraction_target:
            if intent.extraction_target == 'inclusion_criteria':
                structured_results = self._extract_all_inclusion_criteria(study_id)
            elif intent.extraction_target == 'exclusion_criteria':
                structured_results = self._extract_all_exclusion_criteria(study_id)
            elif intent.extraction_target == 'endpoints':
                structured_results = self._extract_all_endpoints(study_id)
            elif intent.extraction_target == 'analyses':
                structured_results = self._extract_all_analyses(study_id)
            elif intent.extraction_target == 'populations':
                structured_results = self._extract_all_populations(study_id)
            elif intent.extraction_target == 'domains':
                structured_results = self._extract_all_domains(study_id)
            else:
                structured_results = self._extract_by_domain(intent.domain_focus, study_id)
        else:
            # General structured query
            structured_results = self._general_structured_query(query, study_id)
        
        # Generate response from structured data
        response = self._generate_structured_response(
            structured_results, intent.extraction_target, query
        )
        
        # Calculate confidence
        confidence_scores['structured'] = self.confidence_scorer.score_structured_results(
            structured_results
        )
        
        # Calculate completeness
        completeness_score = self._calculate_completeness(structured_results, intent)
        
        return HybridResult(
            query=query,
            query_type=QueryType.EXHAUSTIVE,
            primary_response=response,
            structured_data=structured_results,
            confidence_scores=confidence_scores,
            completeness_score=completeness_score,
            sources=all_sources
        )
    
    def _query_vector_store(self, query: str, intent: QueryIntent,
                          study_id: Optional[str], top_k: int) -> HybridResult:
        """Query vector store for semantic search."""
        print("   🔍 Querying vector store...")
        
        # Use existing RAG system with smart routing
        rag_result = self.rag_system.query_with_smart_routing(
            query_text=query,
            study_id=study_id,
            top_k=top_k
        )
        
        response = rag_result.get('combined_response', 'No response generated')
        tier_results = rag_result.get('tier_results', {})
        
        # Extract sources
        sources = []
        confidence_scores = {}
        
        for tier_name, tier_data in tier_results.items():
            if isinstance(tier_data, dict) and 'response' in tier_data:
                sources.append({
                    'tier': tier_name,
                    'response_preview': tier_data['response'][:200] + '...',
                    'source_type': 'semantic'
                })
                confidence_scores[tier_name] = 0.8  # Default semantic confidence
        
        return HybridResult(
            query=query,
            query_type=QueryType.SEMANTIC,
            primary_response=response,
            semantic_results=tier_results,
            confidence_scores=confidence_scores,
            sources=sources
        )
    
    def _query_knowledge_graph(self, query: str, intent: QueryIntent,
                             study_id: Optional[str]) -> HybridResult:
        """Query knowledge graph for relationship queries."""
        print("   🕸️  Querying knowledge graph...")
        
        if not self.graph_engine:
            # Fallback to semantic search if no graph engine
            return self._query_vector_store(query, intent, study_id, 5)
        
        # Convert natural language query to graph query
        graph_query = self._generate_graph_query(query, intent, study_id)
        
        # Execute graph query
        graph_results = self.graph_engine.query(graph_query)
        
        # Generate response from graph results
        response = self._generate_graph_response(graph_results, query)
        
        confidence_scores = {'graph': 0.9}  # Graph queries are typically high confidence
        
        return HybridResult(
            query=query,
            query_type=QueryType.GRAPH,
            primary_response=response,
            graph_results=graph_results,
            confidence_scores=confidence_scores,
            sources=[{'source_type': 'knowledge_graph', 'results_count': len(graph_results)}]
        )
    
    def _process_hybrid_query(self, query: str, intent: QueryIntent,
                            study_id: Optional[str], top_k: int) -> HybridResult:
        """Process query using multiple approaches and merge results."""
        print("   🔄 Processing hybrid query...")
        
        # Get results from multiple sources
        structured_result = self._query_structured_data(query, intent, study_id)
        semantic_result = self._query_vector_store(query, intent, study_id, top_k)
        
        # Optional graph query
        graph_result = None
        if self.graph_engine and intent.query_type == QueryType.GRAPH:
            graph_result = self._query_knowledge_graph(query, intent, study_id)
        
        # Merge and validate results
        merged_result = self._merge_and_validate(
            structured_result, semantic_result, graph_result
        )
        
        return merged_result
    
    def _merge_and_validate(self, structured_result: HybridResult,
                          semantic_result: HybridResult,
                          graph_result: Optional[HybridResult] = None) -> HybridResult:
        """Merge results from different query approaches."""
        
        # Combine responses intelligently
        response_parts = []
        
        # Start with structured data if available and comprehensive
        if (structured_result.structured_data and 
            structured_result.completeness_score > 0.7):
            response_parts.append("=== COMPREHENSIVE EXTRACTION ===")
            response_parts.append(structured_result.primary_response)
        
        # Add semantic context
        if semantic_result.primary_response:
            response_parts.append("=== CONTEXTUAL INFORMATION ===")
            response_parts.append(semantic_result.primary_response)
        
        # Add graph relationships if available
        if graph_result and graph_result.primary_response:
            response_parts.append("=== RELATIONSHIPS & CROSS-REFERENCES ===")
            response_parts.append(graph_result.primary_response)
        
        combined_response = "\n\n".join(response_parts)
        
        # Merge confidence scores
        combined_confidence = {}
        for result in [structured_result, semantic_result, graph_result]:
            if result and result.confidence_scores:
                combined_confidence.update(result.confidence_scores)

        # Combine sources
        combined_sources = []
        for result in [structured_result, semantic_result, graph_result]:
            if result and result.sources:
                combined_sources.extend(result.sources)
        
        # Calculate overall completeness
        overall_completeness = max(
            structured_result.completeness_score,
            semantic_result.completeness_score if semantic_result else 0,
            graph_result.completeness_score if graph_result else 0
        )
        
        return HybridResult(
            query=structured_result.query,
            query_type=QueryType.HYBRID,
            primary_response=combined_response,
            structured_data=structured_result.structured_data,
            semantic_results=semantic_result.semantic_results,
            graph_results=graph_result.graph_results if graph_result else None,
            confidence_scores=combined_confidence,
            completeness_score=overall_completeness,
            sources=combined_sources
        )
    
    # Structured data extraction methods
    def _extract_all_inclusion_criteria(self, study_id: str) -> Dict[str, Any]:
        """Extract all inclusion criteria from structured data."""
        criteria = self.structured_store.query(
            "SELECT * FROM inclusion_criteria WHERE study_id = ?", 
            [study_id]
        )
        
        return {
            'inclusion_criteria': criteria,
            'total_count': len(criteria),
            'extraction_complete': len(criteria) > 0,
            'last_updated': datetime.datetime.now().isoformat()
        }
    
    def _extract_all_exclusion_criteria(self, study_id: str) -> Dict[str, Any]:
        """Extract all exclusion criteria from structured data."""
        criteria = self.structured_store.query(
            "SELECT * FROM exclusion_criteria WHERE study_id = ?",
            [study_id]
        )
        
        return {
            'exclusion_criteria': criteria,
            'total_count': len(criteria),
            'extraction_complete': len(criteria) > 0,
            'last_updated': datetime.datetime.now().isoformat()
        }
    
    def _extract_all_endpoints(self, study_id: str) -> Dict[str, Any]:
        """Extract all endpoints from structured data."""
        primary_endpoints = self.structured_store.query(
            "SELECT * FROM endpoints WHERE study_id = ? AND type = 'primary'",
            [study_id]
        )
        
        secondary_endpoints = self.structured_store.query(
            "SELECT * FROM endpoints WHERE study_id = ? AND type = 'secondary'",
            [study_id]
        )
        
        return {
            'primary_endpoints': primary_endpoints,
            'secondary_endpoints': secondary_endpoints,
            'total_primary': len(primary_endpoints),
            'total_secondary': len(secondary_endpoints),
            'extraction_complete': len(primary_endpoints) > 0,
            'last_updated': datetime.datetime.now().isoformat()
        }
    
    def _extract_all_analyses(self, study_id: str) -> Dict[str, Any]:
        """Extract all analysis specifications from structured data."""
        analyses = self.structured_store.query(
            "SELECT * FROM analysis_specifications WHERE study_id = ?",
            [study_id]
        )
        
        # Group by analysis type
        grouped_analyses = {}
        for analysis in analyses:
            analysis_type = analysis.get('analysis_type', 'other')
            if analysis_type not in grouped_analyses:
                grouped_analyses[analysis_type] = []
            grouped_analyses[analysis_type].append(analysis)
        
        return {
            'all_analyses': analyses,
            'analyses_by_type': grouped_analyses,
            'total_count': len(analyses),
            'analysis_types': list(grouped_analyses.keys()),
            'extraction_complete': len(analyses) > 0,
            'last_updated': datetime.datetime.now().isoformat()
        }
    
    def _extract_all_populations(self, study_id: str) -> Dict[str, Any]:
        """Extract all analysis populations from structured data."""
        populations = self.structured_store.query(
            "SELECT * FROM analysis_populations WHERE study_id = ?",
            [study_id]
        )
        
        return {
            'analysis_populations': populations,
            'total_count': len(populations),
            'population_names': [p.get('name') for p in populations],
            'extraction_complete': len(populations) > 0,
            'last_updated': datetime.datetime.now().isoformat()
        }
    
    def _extract_all_domains(self, study_id: str) -> Dict[str, Any]:
        """Extract all SDTM domains from structured data."""
        domains = self.structured_store.query(
            "SELECT * FROM sdtm_domains WHERE study_id = ?",
            [study_id]
        )
        
        # Group by domain type
        domain_info = {}
        for domain in domains:
            domain_name = domain.get('domain_name', 'unknown')
            domain_info[domain_name] = domain
        
        return {
            'sdtm_domains': domains,
            'domain_details': domain_info,
            'total_count': len(domains),
            'domain_names': list(domain_info.keys()),
            'extraction_complete': len(domains) > 0,
            'last_updated': datetime.datetime.now().isoformat()
        }
    
    def _extract_by_domain(self, domain_focus: str, study_id: str) -> Dict[str, Any]:
        """Extract data by domain focus."""
        if domain_focus == 'protocol':
            return {
                **self._extract_all_inclusion_criteria(study_id),
                **self._extract_all_exclusion_criteria(study_id),
                **self._extract_all_endpoints(study_id)
            }
        elif domain_focus == 'sap':
            return {
                **self._extract_all_analyses(study_id),
                **self._extract_all_populations(study_id)
            }
        elif domain_focus == 'define_xml':
            return self._extract_all_domains(study_id)
        else:
            return self._general_structured_query(f"Extract {domain_focus} information", study_id)
    
    def _general_structured_query(self, query: str, study_id: str) -> Dict[str, Any]:
        """General structured query when no specific extraction target."""
        # This would implement a more general extraction based on query content
        # For now, return a summary of available structured data
        
        return {
            'available_extractions': [
                'inclusion_criteria',
                'exclusion_criteria', 
                'primary_endpoints',
                'secondary_endpoints',
                'analysis_specifications',
                'analysis_populations',
                'sdtm_domains'
            ],
            'message': 'Multiple extraction types available. Please specify target.',
            'extraction_complete': False
        }
    
    def _generate_structured_response(self, structured_data: Dict[str, Any],
                                    extraction_target: str, query: str) -> str:
        """Generate human-readable response from structured data."""
        
        if not structured_data or not structured_data.get('extraction_complete', False):
            return f"No structured data found for {extraction_target or 'the requested information'}."
        
        response_parts = []
        
        # Handle different extraction targets
        if extraction_target == 'inclusion_criteria':
            criteria = structured_data.get('inclusion_criteria', [])
            response_parts.append(f"INCLUSION CRITERIA (Total: {len(criteria)})")
            for i, criterion in enumerate(criteria, 1):
                response_parts.append(f"{i}. {criterion.get('description', 'No description')}")
        
        elif extraction_target == 'exclusion_criteria':
            criteria = structured_data.get('exclusion_criteria', [])
            response_parts.append(f"EXCLUSION CRITERIA (Total: {len(criteria)})")
            for i, criterion in enumerate(criteria, 1):
                response_parts.append(f"{i}. {criterion.get('description', 'No description')}")
        
        elif extraction_target == 'endpoints':
            primary = structured_data.get('primary_endpoints', [])
            secondary = structured_data.get('secondary_endpoints', [])
            
            response_parts.append(f"PRIMARY ENDPOINTS (Total: {len(primary)})")
            for i, endpoint in enumerate(primary, 1):
                response_parts.append(f"{i}. {endpoint.get('description', 'No description')}")
                if endpoint.get('statistical_method'):
                    response_parts.append(f"   Statistical Method: {endpoint['statistical_method']}")
            
            if secondary:
                response_parts.append(f"\nSECONDARY ENDPOINTS (Total: {len(secondary)})")
                for i, endpoint in enumerate(secondary, 1):
                    response_parts.append(f"{i}. {endpoint.get('description', 'No description')}")
        
        elif extraction_target == 'analyses':
            analyses = structured_data.get('all_analyses', [])
            analyses_by_type = structured_data.get('analyses_by_type', {})
            
            response_parts.append(f"ANALYSIS SPECIFICATIONS (Total: {len(analyses)})")
            
            for analysis_type, type_analyses in analyses_by_type.items():
                response_parts.append(f"\n{analysis_type.upper()} ANALYSES:")
                for i, analysis in enumerate(type_analyses, 1):
                    response_parts.append(f"{i}. {analysis.get('description', 'No description')}")
                    if analysis.get('statistical_method'):
                        response_parts.append(f"   Method: {analysis['statistical_method']}")
                    if analysis.get('population'):
                        response_parts.append(f"   Population: {analysis['population']}")
        
        elif extraction_target == 'populations':
            populations = structured_data.get('analysis_populations', [])
            response_parts.append(f"ANALYSIS POPULATIONS (Total: {len(populations)})")
            for i, pop in enumerate(populations, 1):
                response_parts.append(f"{i}. {pop.get('name', 'Unnamed Population')}")
                if pop.get('definition'):
                    response_parts.append(f"   Definition: {pop['definition']}")
                if pop.get('criteria'):
                    response_parts.append(f"   Criteria: {pop['criteria']}")
        
        elif extraction_target == 'domains':
            domains = structured_data.get('sdtm_domains', [])
            response_parts.append(f"SDTM DOMAINS (Total: {len(domains)})")
            for domain in domains:
                domain_name = domain.get('domain_name', 'Unknown')
                description = domain.get('description', 'No description')
                variable_count = len(domain.get('variables', []))
                response_parts.append(f"• {domain_name}: {description} ({variable_count} variables)")
        
        else:
            # General response
            response_parts.append("STRUCTURED DATA SUMMARY:")
            for key, value in structured_data.items():
                if isinstance(value, list):
                    response_parts.append(f"• {key}: {len(value)} items")
                elif isinstance(value, dict):
                    response_parts.append(f"• {key}: {len(value)} entries")
                else:
                    response_parts.append(f"• {key}: {value}")
        
        return "\n".join(response_parts)
    
    def _generate_graph_query(self, query: str, intent: QueryIntent, study_id: str) -> str:
        """Convert natural language query to graph query."""
        # This is a simplified example - in practice, you'd use more sophisticated NL to Cypher conversion
        
        if 'relationship' in query.lower() or 'related' in query.lower():
            return f"""
                MATCH (a)-[r]-(b) 
                WHERE a.study_id = '{study_id}' 
                RETURN a, r, b 
                LIMIT 20
            """
        elif 'endpoint' in query.lower() and 'analysis' in query.lower():
            return f"""
                MATCH (e:Endpoint)-[r:ANALYZED_BY]-(a:Analysis)
                WHERE e.study_id = '{study_id}'
                RETURN e.name, a.method, a.population
            """
        else:
            return f"""
                MATCH (n) 
                WHERE n.study_id = '{study_id}' 
                RETURN n 
                LIMIT 10
            """
    
    def _generate_graph_response(self, graph_results: List[Dict], query: str) -> str:
        """Generate response from graph query results."""
        if not graph_results:
            return "No relationships found in the knowledge graph."
        
        response_parts = ["KNOWLEDGE GRAPH RELATIONSHIPS:"]
        
        for i, result in enumerate(graph_results[:10], 1):  # Limit to top 10
            if 'relationship' in result:
                response_parts.append(f"{i}. {result['relationship']}")
            else:
                # Format based on available data
                response_parts.append(f"{i}. {str(result)[:100]}...")
        
        if len(graph_results) > 10:
            response_parts.append(f"\n... and {len(graph_results) - 10} more relationships")
        
        return "\n".join(response_parts)
    
    def _calculate_completeness(self, structured_data: Dict[str, Any], 
                              intent: QueryIntent) -> float:
        """Calculate completeness score for extraction."""
        if not structured_data:
            return 0.0
        
        # Check if extraction found data
        extraction_complete = structured_data.get('extraction_complete', False)
        if not extraction_complete:
            return 0.0
        
        # Calculate based on extraction target
        if intent.extraction_target in ['inclusion_criteria', 'exclusion_criteria']:
            count = structured_data.get('total_count', 0)
            # Expect at least 3 criteria for completeness
            return min(1.0, count / 3.0)
        
        elif intent.extraction_target == 'endpoints':
            primary_count = structured_data.get('total_primary', 0)
            secondary_count = structured_data.get('total_secondary', 0)
            # Expect at least 1 primary endpoint
            return min(1.0, (primary_count + secondary_count * 0.5) / 1.0)
        
        elif intent.extraction_target == 'analyses':
            count = structured_data.get('total_count', 0)
            # Expect at least 5 analyses for completeness
            return min(1.0, count / 5.0)
        
        else:
            # General completeness based on data presence
            return 0.8 if structured_data else 0.0
    
    def _validate_results(self, result: HybridResult, study_id: str) -> Dict[str, Any]:
        """Validate query results for consistency and completeness."""
        validation = {
            'consistency_check': True,
            'completeness_verified': result.completeness_score > 0.7,
            'cross_validation_passed': True,
            'confidence_acceptable': all(score > 0.5 for score in result.confidence_scores.values()) if result.confidence_scores else False,
            'warnings': []
        }
        
        # Check for inconsistencies between structured and semantic results
        if result.structured_data and result.semantic_results:
            # This would implement more sophisticated cross-validation
            # For now, just check basic consistency
            structured_count = result.structured_data.get('total_count', 0)
            if structured_count == 0:
                validation['warnings'].append("Structured extraction found no data but semantic search may have found relevant information")
        
        # Check completeness threshold
        if result.completeness_score < 0.5:
            validation['warnings'].append(f"Low completeness score: {result.completeness_score:.2f}")
            validation['completeness_verified'] = False
        
        return validation


# Example usage and integration
def create_hybrid_processor(rag_system: TieredRAGSystem) -> HybridQueryProcessor:
    """Factory function to create hybrid processor with dependencies."""
    
    # Mock structured store for example - replace with actual implementation
    class MockStructuredStore:
        def query(self, sql: str, params: List[str] = None):
            # This would connect to actual structured database
            return []
    
    structured_store = MockStructuredStore()
    
    # Optional graph engine - can be None for fallback to semantic search
    graph_engine = None  # Would initialize GraphQueryEngine here
    
    return HybridQueryProcessor(
        rag_system=rag_system,
        structured_store=structured_store,
        graph_engine=graph_engine
    ) 
