# structured_extractors/protocol_extractor.py
"""
Enhanced Protocol Extractor - Comprehensive structured extraction from clinical protocol documents

Extracts key structured information including:
- Protocol summary and estimands
- Inclusion/exclusion criteria
- Primary and secondary endpoints  
- Study design elements
- Treatment arms and dosing
- Study populations
- Schedule of assessments and planned procedures
- Discontinuation criteria
- Adverse events handling
- Statistical analyses
- Pharmacokinetics
"""

import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from llama_index.core import Document, Settings
from .base_extractor import BaseExtractor, StructuredData, ExtractionResult


@dataclass
class ProtocolData(StructuredData):
    """Enhanced structured data extracted from protocol documents."""
    protocol_summary: Dict[str, Any]
    estimands: List[Dict[str, Any]]
    study_design: Dict[str, Any]
    schedule_of_assessments: List[Dict[str, Any]]
    planned_procedures: List[Dict[str, Any]]
    inclusion_criteria: List[Dict[str, Any]]
    exclusion_criteria: List[Dict[str, Any]]
    primary_endpoints: List[Dict[str, Any]]
    secondary_endpoints: List[Dict[str, Any]]
    treatment_arms: List[Dict[str, Any]]
    study_populations: List[Dict[str, Any]]
    dosing_regimens: List[Dict[str, Any]]
    discontinuation: List[Dict[str, Any]]
    adverse_events: Dict[str, Any]
    statistical_analyses: List[Dict[str, Any]]
    pharmacokinetics: List[Dict[str, Any]]


class ProtocolExtractor(BaseExtractor):
    """Enhanced extractor for structured information from protocol documents."""
    
    def __init__(self):
        super().__init__()
        self.document_type = "protocol"
        
        # Patterns for identifying sections
        self.section_patterns = {
            'protocol_summary': [
                r'(?:protocol\s+)?summary',
                r'study\s+overview',
                r'executive\s+summary',
                r'synopsis'
            ],
            'estimands': [
                r'estimands?',
                r'treatment\s+effect',
                r'causal\s+inference'
            ],
            'inclusion_criteria': [
                r'inclusion\s+criteria',
                r'eligible\s+(?:patients|subjects)',
                r'entry\s+criteria',
                r'patient\s+selection\s+criteria'
            ],
            'exclusion_criteria': [
                r'exclusion\s+criteria',
                r'excluded\s+(?:patients|subjects)',
                r'exclusionary\s+criteria'
            ],
            'primary_endpoints': [
                r'primary\s+(?:endpoint|outcome|objective)',
                r'primary\s+efficacy\s+(?:endpoint|measure)'
            ],
            'secondary_endpoints': [
                r'secondary\s+(?:endpoint|outcome|objective)',
                r'secondary\s+efficacy\s+(?:endpoint|measure)'
            ],
            'study_design': [
                r'study\s+design',
                r'trial\s+design',
                r'experimental\s+design'
            ],
            'schedule_of_assessments': [
                r'schedule\s+of\s+(?:assessments|events)',
                r'visit\s+schedule',
                r'study\s+schedule',
                r'assessment\s+schedule'
            ],
            'planned_procedures': [
                r'planned\s+procedures',
                r'study\s+procedures',
                r'protocol\s+procedures'
            ],
            'discontinuation': [
                r'discontinuation\s+criteria',
                r'withdrawal\s+criteria',
                r'stopping\s+rules',
                r'early\s+termination'
            ],
            'adverse_events': [
                r'adverse\s+events?',
                r'safety\s+monitoring',
                r'toxicity\s+management',
                r'side\s+effects'
            ],
            'statistical_analyses': [
                r'statistical\s+(?:analysis|methods)',
                r'data\s+analysis',
                r'statistical\s+plan'
            ],
            'pharmacokinetics': [
                r'pharmacokinetics?',
                r'pharmacodynamics?',
                r'pk/pd',
                r'drug\s+concentration'
            ]
        }
        
        # Patterns for extracting individual items
        self.item_patterns = {
            'criterion_item': r'(?:\d+\.?\s*|[a-z]\)\s*|•\s*)(.*?)(?=\n\s*(?:\d+\.|\w\)|•|$))',
            'endpoint_item': r'(?:\d+\.?\s*|[a-z]\)\s*|•\s*)(.*?)(?=\n\s*(?:\d+\.|\w\)|•|$))',
            'numbered_list': r'(\d+\.?\s+)(.*?)(?=\n\s*\d+\.|\n\s*[a-z]\)|\n\s*•|$)',
            'bulleted_list': r'(•\s+|[*]\s+)(.*?)(?=\n\s*•|\n\s*[*]|\n\s*\d+\.|\n\s*[a-z]\)|$)',
            'table_row': r'\|([^|]+)\|([^|]+)\|([^|]*)\|?'
        }
    
    def extract(self, document: Document) -> ExtractionResult:
        """Extract comprehensive structured data from protocol document."""
        try:
            text = document.text
            
            # Extract all types of information
            protocol_summary = self._extract_protocol_summary(text)
            estimands = self._extract_estimands(text)
            study_design = self._extract_study_design(text)
            schedule_of_assessments = self._extract_schedule_of_assessments(text)
            planned_procedures = self._extract_planned_procedures(text)
            inclusion_criteria = self._extract_inclusion_criteria(text)
            exclusion_criteria = self._extract_exclusion_criteria(text)
            primary_endpoints = self._extract_primary_endpoints(text)
            secondary_endpoints = self._extract_secondary_endpoints(text)
            treatment_arms = self._extract_treatment_arms(text)
            study_populations = self._extract_study_populations(text)
            dosing_regimens = self._extract_dosing_regimens(text)
            discontinuation = self._extract_discontinuation(text)
            adverse_events = self._extract_adverse_events(text)
            statistical_analyses = self._extract_statistical_analyses(text)
            pharmacokinetics = self._extract_pharmacokinetics(text)
            
            # Create structured data object
            protocol_data = ProtocolData(
                document_id=document.metadata.get('file_id', 'unknown'),
                document_type='protocol',
                extraction_timestamp=self._get_timestamp(),
                confidence_score=self._calculate_confidence([
                    inclusion_criteria, exclusion_criteria, primary_endpoints,
                    study_design, treatment_arms
                ]),
                protocol_summary=protocol_summary,
                estimands=estimands,
                study_design=study_design,
                schedule_of_assessments=schedule_of_assessments,
                planned_procedures=planned_procedures,
                inclusion_criteria=inclusion_criteria,
                exclusion_criteria=exclusion_criteria,
                primary_endpoints=primary_endpoints,
                secondary_endpoints=secondary_endpoints,
                treatment_arms=treatment_arms,
                study_populations=study_populations,
                dosing_regimens=dosing_regimens,
                discontinuation=discontinuation,
                adverse_events=adverse_events,
                statistical_analyses=statistical_analyses,
                pharmacokinetics=pharmacokinetics
            )
            
            # Calculate completeness
            completeness = self._calculate_completeness(protocol_data)
            
            return ExtractionResult(
                success=True,
                data=protocol_data,
                completeness_score=completeness,
                extraction_metadata={
                    'sections_found': self._get_sections_found(protocol_data),
                    'total_criteria': len(inclusion_criteria) + len(exclusion_criteria),
                    'total_endpoints': len(primary_endpoints) + len(secondary_endpoints),
                    'total_procedures': len(planned_procedures),
                    'total_assessments': len(schedule_of_assessments)
                }
            )
            
        except Exception as e:
            return ExtractionResult(
                success=False,
                error=str(e),
                data=None,
                completeness_score=0.0
            )
    
    def _extract_protocol_summary(self, text: str) -> Dict[str, Any]:
        """Extract protocol summary information."""
        summary = {
            'title': 'unknown',
            'sponsor': 'unknown',
            'phase': 'unknown',
            'indication': 'unknown',
            'objectives': [],
            'sample_size': 'unknown',
            'duration': 'unknown'
        }
        
        # Extract title
        title_patterns = [
            r'(?:protocol\s+)?title[:\s]+(.*?)(?:\n|$)',
            r'study\s+title[:\s]+(.*?)(?:\n|$)',
            r'^(.{20,100})(?:\n|$)'  # First substantial line
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                summary['title'] = self._clean_text(match.group(1))
                break
        
        # Extract sponsor
        sponsor_match = re.search(r'sponsor[:\s]+(.*?)(?:\n|$)', text, re.IGNORECASE)
        if sponsor_match:
            summary['sponsor'] = self._clean_text(sponsor_match.group(1))
        
        # Extract phase
        summary['phase'] = self._extract_study_phase(text)
        
        # Extract indication
        indication_patterns = [
            r'indication[:\s]+(.*?)(?:\n|$)',
            r'disease[:\s]+(.*?)(?:\n|$)',
            r'condition[:\s]+(.*?)(?:\n|$)'
        ]
        
        for pattern in indication_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                summary['indication'] = self._clean_text(match.group(1))
                break
        
        # Extract objectives from summary section
        summary_section = self._find_section(text, 'protocol_summary')
        if summary_section:
            objectives = self._extract_objectives(summary_section)
            summary['objectives'] = objectives
        
        summary['sample_size'] = self._extract_sample_size(text)
        summary['duration'] = self._extract_study_duration(text)
        
        return summary
    
    def _extract_estimands(self, text: str) -> List[Dict[str, Any]]:
        """Extract estimands information."""
        estimands = []
        
        section_text = self._find_section(text, 'estimands')
        if not section_text:
            return estimands
        
        # Look for estimand components
        estimand_patterns = [
            r'(?:population|target\s+population)[:\s]+(.*?)(?=\n|treatment|outcome)',
            r'(?:treatment|intervention)[:\s]+(.*?)(?=\n|outcome|population)',
            r'(?:outcome|endpoint)[:\s]+(.*?)(?=\n|population|treatment)',
            r'(?:handling|strategy)[:\s]+(.*?)(?=\n|population|treatment|outcome)'
        ]
        
        for i, pattern in enumerate(estimand_patterns):
            matches = re.finditer(pattern, section_text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                estimand = {
                    'id': f"estimand_{i+1}",
                    'component': ['population', 'treatment', 'outcome', 'handling'][i % 4],
                    'description': self._clean_text(match.group(1)),
                    'order': i + 1
                }
                estimands.append(estimand)
        
        return estimands
    
    def _extract_schedule_of_assessments(self, text: str) -> List[Dict[str, Any]]:
        """Extract schedule of assessments."""
        assessments = []
        
        section_text = self._find_section(text, 'schedule_of_assessments')
        if not section_text:
            return assessments
        
        # Look for table-like structures
        table_matches = re.findall(self.item_patterns['table_row'], section_text)
        
        for i, match in enumerate(table_matches):
            if len(match) >= 2:
                assessment = {
                    'id': f"assessment_{i+1}",
                    'visit': self._clean_text(match[0]),
                    'timepoint': self._clean_text(match[1]),
                    'procedures': self._clean_text(match[2]) if len(match) > 2 else '',
                    'order': i + 1
                }
                assessments.append(assessment)
        
        # If no table found, look for visit descriptions
        if not assessments:
            visit_patterns = [
                r'visit\s+(\d+)[:\s]+(.*?)(?=visit\s+\d+|$)',
                r'day\s+(\d+)[:\s]+(.*?)(?=day\s+\d+|$)',
                r'week\s+(\d+)[:\s]+(.*?)(?=week\s+\d+|$)'
            ]
            
            for pattern in visit_patterns:
                matches = re.finditer(pattern, section_text, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    assessment = {
                        'id': f"visit_{match.group(1)}",
                        'visit': f"Visit {match.group(1)}",
                        'timepoint': f"Day/Week {match.group(1)}",
                        'procedures': self._clean_text(match.group(2)),
                        'order': int(match.group(1))
                    }
                    assessments.append(assessment)
        
        return assessments
    
    def _extract_planned_procedures(self, text: str) -> List[Dict[str, Any]]:
        """Extract planned procedures."""
        procedures = []
        
        section_text = self._find_section(text, 'planned_procedures')
        if not section_text:
            return procedures
        
        # Extract procedure lists
        procedure_texts = []
        
        # Try numbered list first
        numbered_matches = re.findall(self.item_patterns['numbered_list'], section_text, re.DOTALL)
        if numbered_matches:
            procedure_texts = [match[1].strip() for match in numbered_matches]
        else:
            # Try bulleted list
            bulleted_matches = re.findall(self.item_patterns['bulleted_list'], section_text, re.DOTALL)
            if bulleted_matches:
                procedure_texts = [match[1].strip() for match in bulleted_matches]
        
        # Structure the procedures
        for i, procedure_text in enumerate(procedure_texts):
            if len(procedure_text.strip()) < 10:
                continue
            
            procedure = {
                'id': f"procedure_{i+1}",
                'description': self._clean_text(procedure_text),
                'category': self._categorize_procedure(procedure_text),
                'frequency': self._extract_frequency(procedure_text),
                'timing': self._extract_timing(procedure_text),
                'order': i + 1
            }
            procedures.append(procedure)
        
        return procedures
    
    def _extract_discontinuation(self, text: str) -> List[Dict[str, Any]]:
        """Extract discontinuation criteria."""
        discontinuation_criteria = []
        
        section_text = self._find_section(text, 'discontinuation')
        if not section_text:
            return discontinuation_criteria
        
        # Extract criteria using similar pattern to inclusion/exclusion
        criteria_texts = []
        
        numbered_matches = re.findall(self.item_patterns['numbered_list'], section_text, re.DOTALL)
        if numbered_matches:
            criteria_texts = [match[1].strip() for match in numbered_matches]
        else:
            bulleted_matches = re.findall(self.item_patterns['bulleted_list'], section_text, re.DOTALL)
            if bulleted_matches:
                criteria_texts = [match[1].strip() for match in bulleted_matches]
        
        for i, criterion_text in enumerate(criteria_texts):
            if len(criterion_text.strip()) < 10:
                continue
            
            criterion = {
                'id': f"discontinuation_{i+1}",
                'description': self._clean_text(criterion_text),
                'category': self._categorize_discontinuation(criterion_text),
                'severity': self._extract_severity(criterion_text),
                'order': i + 1
            }
            discontinuation_criteria.append(criterion)
        
        return discontinuation_criteria
    
    def _extract_adverse_events(self, text: str) -> Dict[str, Any]:
        """Extract adverse events handling information."""
        adverse_events = {
            'reporting_procedures': 'unknown',
            'severity_grading': 'unknown',
            'causality_assessment': 'unknown',
            'follow_up_procedures': 'unknown',
            'serious_ae_criteria': []
        }
        
        section_text = self._find_section(text, 'adverse_events')
        if not section_text:
            return adverse_events
        
        # Extract reporting procedures
        reporting_match = re.search(
            r'reporting\s+(?:procedures|process)[:\s]+(.*?)(?=\n\s*[A-Z]|$)',
            section_text,
            re.IGNORECASE | re.DOTALL
        )
        if reporting_match:
            adverse_events['reporting_procedures'] = self._clean_text(reporting_match.group(1))
        
        # Extract severity grading
        if re.search(r'CTCAE|common\s+terminology\s+criteria', section_text, re.IGNORECASE):
            adverse_events['severity_grading'] = 'CTCAE'
        elif re.search(r'WHO\s+toxicity', section_text, re.IGNORECASE):
            adverse_events['severity_grading'] = 'WHO'
        
        # Extract serious AE criteria
        sae_patterns = [
            r'death',
            r'life.?threatening',
            r'hospitalization',
            r'disability',
            r'congenital\s+anomaly',
            r'medically\s+important'
        ]
        
        for pattern in sae_patterns:
            if re.search(pattern, section_text, re.IGNORECASE):
                adverse_events['serious_ae_criteria'].append(pattern.replace(r'\.?', ''))
        
        return adverse_events
    
    def _extract_statistical_analyses(self, text: str) -> List[Dict[str, Any]]:
        """Extract statistical analysis plans."""
        analyses = []
        
        section_text = self._find_section(text, 'statistical_analyses')
        if not section_text:
            return analyses
        
        # Look for analysis descriptions
        analysis_patterns = [
            r'primary\s+analysis[:\s]+(.*?)(?=secondary|$)',
            r'secondary\s+analysis[:\s]+(.*?)(?=primary|$)',
            r'(?:descriptive|exploratory)\s+analysis[:\s]+(.*?)(?=\n\s*[A-Z]|$)'
        ]
        
        for i, pattern in enumerate(analysis_patterns):
            matches = re.finditer(pattern, section_text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                analysis = {
                    'id': f"analysis_{i+1}",
                    'type': ['primary', 'secondary', 'exploratory'][i % 3],
                    'description': self._clean_text(match.group(1)),
                    'statistical_method': self._extract_statistical_method(match.group(1)),
                    'significance_level': self._extract_significance_level(match.group(1)),
                    'order': i + 1
                }
                analyses.append(analysis)
        
        return analyses
    
    def _extract_pharmacokinetics(self, text: str) -> List[Dict[str, Any]]:
        """Extract pharmacokinetic information."""
        pk_data = []
        
        section_text = self._find_section(text, 'pharmacokinetics')
        if not section_text:
            return pk_data
        
        # Look for PK parameters
        pk_parameters = [
            r'(?:Cmax|maximum\s+concentration)[:\s]+(.*?)(?=\n|AUC)',
            r'(?:AUC|area\s+under\s+curve)[:\s]+(.*?)(?=\n|Cmax)',
            r'(?:Tmax|time\s+to\s+maximum)[:\s]+(.*?)(?=\n|half)',
            r'(?:T1/2|half.?life)[:\s]+(.*?)(?=\n|clearance)',
            r'(?:clearance|CL)[:\s]+(.*?)(?=\n|volume)'
        ]
        
        for i, pattern in enumerate(pk_parameters):
            matches = re.finditer(pattern, section_text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                pk_param = {
                    'id': f"pk_param_{i+1}",
                    'parameter': ['Cmax', 'AUC', 'Tmax', 'T1/2', 'Clearance'][i % 5],
                    'description': self._clean_text(match.group(1)),
                    'sampling_times': self._extract_sampling_times(match.group(1)),
                    'order': i + 1
                }
                pk_data.append(pk_param)
        
        return pk_data
    
    # Enhanced helper methods
    def _categorize_procedure(self, procedure_text: str) -> str:
        """Categorize a procedure by type."""
        text_lower = procedure_text.lower()
        
        if any(word in text_lower for word in ['blood', 'laboratory', 'lab', 'chemistry', 'hematology']):
            return 'laboratory'
        elif any(word in text_lower for word in ['imaging', 'scan', 'mri', 'ct', 'x-ray', 'ultrasound']):
            return 'imaging'
        elif any(word in text_lower for word in ['biopsy', 'tissue', 'pathology']):
            return 'biopsy'
        elif any(word in text_lower for word in ['questionnaire', 'assessment', 'scale', 'evaluation']):
            return 'assessment'
        elif any(word in text_lower for word in ['ecg', 'ekg', 'cardiac', 'vital signs']):
            return 'cardiac_monitoring'
        else:
            return 'other'
    
    def _extract_timing(self, procedure_text: str) -> str:
        """Extract timing information from procedure text."""
        timing_patterns = [
            r'(?:at|on)\s+(?:day|week|month)\s+(\d+)',
            r'(?:baseline|screening|end\s+of\s+study)',
            r'(?:pre|post).?(?:dose|treatment)',
            r'every\s+(\d+)\s+(?:days?|weeks?|months?)'
        ]
        
        for pattern in timing_patterns:
            match = re.search(pattern, procedure_text, re.IGNORECASE)
            if match:
                return match.group()
        
        return 'not_specified'
    
    def _categorize_discontinuation(self, criterion_text: str) -> str:
        """Categorize discontinuation criterion."""
        text_lower = criterion_text.lower()
        
        if any(word in text_lower for word in ['adverse', 'toxicity', 'safety', 'intolerable']):
            return 'safety'
        elif any(word in text_lower for word in ['progression', 'disease', 'efficacy', 'lack of response']):
            return 'efficacy'
        elif any(word in text_lower for word in ['consent', 'withdrawal', 'patient', 'subject']):
            return 'patient_choice'
        elif any(word in text_lower for word in ['protocol', 'violation', 'deviation', 'non-compliance']):
            return 'protocol_deviation'
        else:
            return 'other'
    
    def _extract_severity(self, criterion_text: str) -> str:
        """Extract severity information."""
        if re.search(r'grade\s+[3-5]|severe|life.?threatening', criterion_text, re.IGNORECASE):
            return 'severe'
        elif re.search(r'grade\s+[1-2]|mild|moderate', criterion_text, re.IGNORECASE):
            return 'mild_moderate'
        else:
            return 'not_specified'
    
    def _extract_significance_level(self, analysis_text: str) -> str:
        """Extract significance level from analysis text."""
        sig_match = re.search(r'(?:α|alpha|significance)\s*=?\s*(0?\.\d+)', analysis_text, re.IGNORECASE)
        if sig_match:
            return sig_match.group(1)
        elif re.search(r'0\.05|5%', analysis_text):
            return '0.05'
        else:
            return 'not_specified'
    
    def _extract_sampling_times(self, pk_text: str) -> List[str]:
        """Extract PK sampling timepoints."""
        times = []
        
        time_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)',
            r'pre.?dose',
            r'post.?dose',
            r'(\d+)\s*(?:minutes?|mins?|min)'
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, pk_text, re.IGNORECASE)
            times.extend(matches)
        
        return times[:10]  # Limit to first 10 timepoints
    
    def _extract_objectives(self, summary_text: str) -> List[str]:
        """Extract study objectives from summary."""
        objectives = []
        
        obj_patterns = [
            r'(?:primary\s+)?objectives?[:\s]+(.*?)(?=secondary|methods|$)',
            r'aims?[:\s]+(.*?)(?=methods|design|$)',
            r'purpose[:\s]+(.*?)(?=methods|design|$)'
        ]
        
        for pattern in obj_patterns:
            match = re.search(pattern, summary_text, re.IGNORECASE | re.DOTALL)
            if match:
                obj_text = self._clean_text(match.group(1))
                # Split by common delimiters
                obj_list = re.split(r'[;,]\s*(?=to\s+)', obj_text)
                objectives.extend([obj.strip() for obj in obj_list if len(obj.strip()) > 10])
        
        return objectives[:5]  # Limit to first 5 objectives
    
    # Inherited and enhanced methods from original
    def _extract_inclusion_criteria(self, text: str) -> List[Dict[str, Any]]:
        """Extract inclusion criteria from protocol text."""
        return self._extract_criteria_list(text, 'inclusion_criteria')
    
    def _extract_exclusion_criteria(self, text: str) -> List[Dict[str, Any]]:
        """Extract exclusion criteria from protocol text."""
        return self._extract_criteria_list(text, 'exclusion_criteria')
    
    def _extract_criteria_list(self, text: str, criteria_type: str) -> List[Dict[str, Any]]:
        """Generic method to extract criteria lists."""
        criteria = []
        
        # Find the relevant section
        section_text = self._find_section(text, criteria_type)
        if not section_text:
            return criteria
        
        # Extract individual criteria using multiple patterns
        criterion_texts = []
        
        # Try numbered list pattern first
        numbered_matches = re.findall(self.item_patterns['numbered_list'], section_text, re.DOTALL | re.IGNORECASE)
        if numbered_matches:
            criterion_texts = [match[1].strip() for match in numbered_matches]
        else:
            # Try bulleted list pattern
            bulleted_matches = re.findall(self.item_patterns['bulleted_list'], section_text, re.DOTALL | re.IGNORECASE)
            if bulleted_matches:
                criterion_texts = [match[1].strip() for match in bulleted_matches]
            else:
                # Try general criterion pattern
                general_matches = re.findall(self.item_patterns['criterion_item'], section_text, re.DOTALL | re.IGNORECASE)
                criterion_texts = [match.strip() for match in general_matches]
        
        # Clean and structure the criteria
        for i, criterion_text in enumerate(criterion_texts):
            if len(criterion_text.strip()) < 10:  # Skip very short items
                continue
                
            criterion = {
                'id': f"{criteria_type}_{i+1}",
                'description': self._clean_text(criterion_text),
                'category': self._categorize_criterion(criterion_text),
                'type': criteria_type.replace('_criteria', ''),
                'order': i + 1
            }
            
            # Extract specific details if possible
            details = self._extract_criterion_details(criterion_text)
            criterion.update(details)
            
            criteria.append(criterion)
        
        return criteria
    
    def _extract_primary_endpoints(self, text: str) -> List[Dict[str, Any]]:
        """Extract primary endpoints from protocol text."""
        return self._extract_endpoints(text, 'primary_endpoints')
    
    def _extract_secondary_endpoints(self, text: str) -> List[Dict[str, Any]]:
        """Extract secondary endpoints from protocol text."""
        return self._extract_endpoints(text, 'secondary_endpoints')
    
    def _extract_endpoints(self, text: str, endpoint_type: str) -> List[Dict[str, Any]]:
        """Generic method to extract endpoints."""
        endpoints = []
        
        # Find the relevant section
        section_text = self._find_section(text, endpoint_type)
        if not section_text:
            return endpoints
        
        # Extract individual endpoints
        endpoint_texts = []
        
        # Try multiple extraction patterns
        numbered_matches = re.findall(self.item_patterns['numbered_list'], section_text, re.DOTALL | re.IGNORECASE)
        if numbered_matches:
            endpoint_texts = [match[1].strip() for match in numbered_matches]
        else:
            bulleted_matches = re.findall(self.item_patterns['bulleted_list'], section_text, re.DOTALL | re.IGNORECASE)
            if bulleted_matches:
                endpoint_texts = [match[1].strip() for match in bulleted_matches]
            else:
                # Single endpoint - take the whole section
                endpoint_texts = [section_text.strip()]
        
        # Structure the endpoints
        for i, endpoint_text in enumerate(endpoint_texts):
            if len(endpoint_text.strip()) < 20:  # Skip very short items
                continue
            
            endpoint = {
                'id': f"{endpoint_type}_{i+1}",
                'description': self._clean_text(endpoint_text),
                'type': endpoint_type.replace('_endpoints', ''),
                'order': i + 1,
                'measurement_method': self._extract_measurement_method(endpoint_text),
                'time_point': self._extract_time_point(endpoint_text),
                'statistical_method': self._extract_statistical_method(endpoint_text)
            }
            
            # Classify endpoint by therapeutic area
            endpoint['category'] = self._classify_endpoint(endpoint_text)
            
            endpoints.append(endpoint)
        
        return endpoints
    
    def _extract_study_design(self, text: str) -> Dict[str, Any]:
        """Extract study design information."""
        design = {
            'design_type': 'unknown',
            'phase': 'unknown',
            'randomization': 'unknown',
            'blinding': 'unknown',
            'control': 'unknown',
            'duration': 'unknown',
            'sample_size': 'unknown'
        }
        
        # Find study design section
        section_text = self._find_section(text, 'study_design')
        if not section_text:
            # Look for design information in the whole document
            section_text = text
        
        # Extract design characteristics
        design['design_type'] = self._extract_design_type(section_text)
        design['phase'] = self._extract_study_phase(section_text)
        design['randomization'] = self._extract_randomization(section_text)
        design['blinding'] = self._extract_blinding(section_text)
        design['control'] = self._extract_control_type(section_text)
        design['duration'] = self._extract_study_duration(section_text)
        design['sample_size'] = self._extract_sample_size(section_text)
        
        return design
    
    def _extract_treatment_arms(self, text: str) -> List[Dict[str, Any]]:
        """Extract treatment arm information."""
        arms = []
        
        # Look for treatment arm patterns
        arm_patterns = [
            r'(?:arm|group)\s*[a-z]?:?\s*(.*?)(?=(?:arm|group)\s*[a-z]?:|$)',
            r'treatment\s*[a-z]?:?\s*(.*?)(?=treatment\s*[a-z]?:|$)',
            r'cohort\s*[a-z]?:?\s*(.*?)(?=cohort\s*[a-z]?:|$)'
        ]
        
        for pattern in arm_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for i, match in enumerate(matches):
                if len(match.strip()) > 20:  # Only substantial descriptions
                    arm = {
                        'id': f"arm_{i+1}",
                        'description': self._clean_text(match),
                        'order': i + 1,
                        'intervention': self._extract_intervention(match),
                        'dose': self._extract_dose(match),
                        'frequency': self._extract_frequency(match)
                    }
                    arms.append(arm)
        
        return arms
    
    def _extract_study_populations(self, text: str) -> List[Dict[str, Any]]:
        """Extract study population definitions."""
        populations = []
        
        # Look for population definition patterns
        pop_patterns = [
            r'(?:safety|efficacy|itt|per\s*protocol|modified\s*itt)\s*(?:population|set|analysis)',
            r'analysis\s*(?:population|set)',
            r'evaluable\s*(?:population|patients)'
        ]
        
        for pattern in pop_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Extract context around the match
                start = max(0, match.start() - 200)
                end = min(len(text), match.end() + 500)
                context = text[start:end]
                
                population = {
                    'name': match.group(),
                    'definition': self._clean_text(context),
                    'type': self._classify_population(match.group())
                }
                populations.append(population)
        
        return populations
    
    def _extract_dosing_regimens(self, text: str) -> List[Dict[str, Any]]:
        """Extract dosing regimen information."""
        regimens = []
        
        # Look for dosing patterns
        dose_patterns = [
            r'(\d+(?:\.\d+)?)\s*(mg|g|mcg|units?)\s*(?:per\s*day|daily|once\s*daily|bid|tid|qid)',
            r'(\d+(?:\.\d+)?)\s*(mg|g|mcg|units?)\s*(?:every\s*\d+\s*hours?|q\d+h)',
            r'starting\s*dose\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*(mg|g|mcg|units?)'
        ]
        
        for pattern in dose_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                dose_value = match.group(1)
                dose_unit = match.group(2)
                
                # Extract context around the dose
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 200)
                context = text[start:end]
                
                regimen = {
                    'dose_value': dose_value,
                    'dose_unit': dose_unit,
                    'frequency': self._extract_frequency_from_context(context),
                    'route': self._extract_route(context),
                    'duration': self._extract_duration_from_context(context),
                    'context': self._clean_text(context)
                }
                regimens.append(regimen)
        
        return regimens
    
    # Helper methods for pattern extraction
    def _find_section(self, text: str, section_type: str) -> Optional[str]:
        """Find and extract a specific section from the text."""
        patterns = self.section_patterns.get(section_type, [])
        
        for pattern in patterns:
            # Look for section headers
            section_match = re.search(
                rf'({pattern})\s*:?\s*(.*?)(?=\n\s*\d+\.?\s*[A-Z]|\n\s*[A-Z][^a-z]*:|\Z)',
                text,
                re.DOTALL | re.IGNORECASE
            )
            
            if section_match:
                return section_match.group(2).strip()
        
        return None
    
    def _categorize_criterion(self, criterion_text: str) -> str:
        """Categorize a criterion by type."""
        text_lower = criterion_text.lower()
        
        if any(word in text_lower for word in ['age', 'years', 'old']):
            return 'demographic'
        elif any(word in text_lower for word in ['diagnosis', 'disease', 'condition']):
            return 'medical_condition'
        elif any(word in text_lower for word in ['consent', 'willing', 'able']):
            return 'consent_ability'
        elif any(word in text_lower for word in ['pregnancy', 'pregnant', 'contraception']):
            return 'reproductive'
        elif any(word in text_lower for word in ['medication', 'drug', 'treatment', 'therapy']):
            return 'concomitant_medication'
        elif any(word in text_lower for word in ['laboratory', 'lab', 'blood', 'hepatic', 'renal']):
            return 'laboratory'
        else:
            return 'other'
    
    def _extract_criterion_details(self, criterion_text: str) -> Dict[str, Any]:
        """Extract specific details from criterion text."""
        details = {}
        
        # Extract age ranges
        age_match = re.search(r'(\d+)\s*(?:to|-)\s*(\d+)\s*years?', criterion_text, re.IGNORECASE)
        if age_match:
            details['age_min'] = int(age_match.group(1))
            details['age_max'] = int(age_match.group(2))
        else:
            age_match = re.search(r'(?:≥|>=|at least)\s*(\d+)\s*years?', criterion_text, re.IGNORECASE)
            if age_match:
                details['age_min'] = int(age_match.group(1))
        
        # Extract laboratory values
        lab_match = re.search(r'([A-Za-z]+)\s*(?:≥|>=|≤|<=|<|>)\s*(\d+(?:\.\d+)?)', criterion_text)
        if lab_match:
            details['lab_parameter'] = lab_match.group(1)
            details['lab_threshold'] = float(lab_match.group(2))
        
        return details
    
    def _extract_measurement_method(self, endpoint_text: str) -> str:
        """Extract measurement method from endpoint description."""
        methods = {
            'RECIST': r'RECIST\s*(?:v?\d+\.\d+)?',
            'CTCAE': r'CTCAE\s*(?:v?\d+\.\d+)?',
            'ECOG': r'ECOG\s*(?:performance\s*status)?',
            'Karnofsky': r'Karnofsky\s*(?:performance\s*status)?',
            'MRI': r'MRI|magnetic\s*resonance',
            'CT': r'CT\s*scan|computed\s*tomography',
            'PET': r'PET\s*scan|positron\s*emission'
        }
        
        for method, pattern in methods.items():
            if re.search(pattern, endpoint_text, re.IGNORECASE):
                return method
        
        return 'not_specified'
    
    def _extract_time_point(self, endpoint_text: str) -> str:
        """Extract time point from endpoint description."""
        time_patterns = [
            r'(?:at|after)\s*(\d+)\s*(?:weeks?|months?|days?)',
            r'(\d+)\s*(?:week|month|day)\s*(?:post|after)',
            r'baseline\s*to\s*(\d+)\s*(?:weeks?|months?)',
            r'end\s*of\s*(?:study|treatment)'
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, endpoint_text, re.IGNORECASE)
            if match:
                if 'baseline' in pattern or 'end' in pattern:
                    return match.group()
                else:
                    return f"{match.group(1)} {match.group().split()[-1]}"
        
        return 'not_specified'
    
    def _extract_statistical_method(self, endpoint_text: str) -> str:
        """Extract statistical method from endpoint description."""
        methods = {
            'log-rank test': r'log.?rank\s*test',
            'chi-square': r'chi.?square|χ²',
            't-test': r't.?test',
            'ANOVA': r'ANOVA|analysis\s*of\s*variance',
            'regression': r'regression\s*(?:analysis)?',
            'Kaplan-Meier': r'Kaplan.?Meier|survival\s*analysis',
            'Cox regression': r'Cox\s*regression|proportional\s*hazards'
        }
        
        for method, pattern in methods.items():
            if re.search(pattern, endpoint_text, re.IGNORECASE):
                return method
        
        return 'not_specified'
    
    def _classify_endpoint(self, endpoint_text: str) -> str:
        """Classify endpoint by therapeutic area or type."""
        text_lower = endpoint_text.lower()
        
        if any(word in text_lower for word in ['survival', 'mortality', 'death']):
            return 'survival'
        elif any(word in text_lower for word in ['response', 'tumor', 'lesion', 'recist']):
            return 'tumor_response'
        elif any(word in text_lower for word in ['quality of life', 'qol', 'functional']):
            return 'quality_of_life'
        elif any(word in text_lower for word in ['safety', 'adverse', 'toxicity', 'side effect', 'reaction']):
            return 'safety'
        elif any(word in text_lower for word in ['biomarker', 'pharmacokinetic', 'pharmacogenomic', 'pk', 'pd']):
            return 'biomarker'
        else:
            return 'efficacy'
    
    def _extract_design_type(self, text: str) -> str:
        """Extract study design type."""
        design_patterns = {
            'randomized controlled trial': r'randomized\s*controlled\s*trial|RCT',
            'open-label': r'open.?label',
            'double-blind': r'double.?blind',
            'single-blind': r'single.?blind',
            'crossover': r'crossover|cross.?over',
            'parallel': r'parallel\s*(?:group|arm)',
            'dose-escalation': r'dose.?escalation',
            'phase I': r'phase\s*I\b|phase\s*1\b',
            'phase II': r'phase\s*II\b|phase\s*2\b',
            'phase III': r'phase\s*III\b|phase\s*3\b'
        }
        
        for design_type, pattern in design_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return design_type
        
        return 'not_specified'
    
    def _extract_study_phase(self, text: str) -> str:
        """Extract study phase."""
        phase_match = re.search(r'phase\s*(I{1,3}|\d)\b', text, re.IGNORECASE)
        if phase_match:
            phase = phase_match.group(1).upper()
            if phase in ['I', '1']:
                return 'Phase I'
            elif phase in ['II', '2']:
                return 'Phase II'
            elif phase in ['III', '3']:
                return 'Phase III'
            elif phase in ['IV', '4']:
                return 'Phase IV'
        return 'not_specified'
    
    def _extract_randomization(self, text: str) -> str:
        """Extract randomization information."""
        if re.search(r'randomized|randomisation', text, re.IGNORECASE):
            # Look for randomization ratio
            ratio_match = re.search(r'(\d+):(\d+)', text)
            if ratio_match:
                return f"randomized {ratio_match.group()}"
            return 'randomized'
        return 'not_randomized'
    
    def _extract_blinding(self, text: str) -> str:
        """Extract blinding information."""
        if re.search(r'double.?blind', text, re.IGNORECASE):
            return 'double-blind'
        elif re.search(r'single.?blind', text, re.IGNORECASE):
            return 'single-blind'
        elif re.search(r'open.?label', text, re.IGNORECASE):
            return 'open-label'
        return 'not_specified'
    
    def _extract_control_type(self, text: str) -> str:
        """Extract control type."""
        if re.search(r'placebo', text, re.IGNORECASE):
            return 'placebo-controlled'
        elif re.search(r'active\s*control', text, re.IGNORECASE):
            return 'active-controlled'
        elif re.search(r'dose\s*control', text, re.IGNORECASE):
            return 'dose-comparison'
        elif re.search(r'historical\s*control', text, re.IGNORECASE):
            return 'historical-controlled'
        return 'not_specified'
    
    def _extract_study_duration(self, text: str) -> str:
        """Extract study duration."""
        duration_patterns = [
            r'(?:study\s*duration|treatment\s*period).*?(\d+)\s*(?:weeks?|months?|years?)',
            r'(\d+)\s*(?:week|month|year)\s*(?:study|trial|treatment)',
            r'follow.?up.*?(\d+)\s*(?:weeks?|months?|years?)'
        ]
        
        for pattern in duration_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group()
        
        return 'not_specified'
    
    def _extract_sample_size(self, text: str) -> str:
        """Extract planned sample size."""
        size_patterns = [
            r'(?:sample\s*size|enroll|recruit).*?(\d+)\s*(?:patients?|subjects?)',
            r'(\d+)\s*(?:patients?|subjects?)\s*(?:will\s*be|to\s*be)\s*(?:enrolled|recruited)',
            r'target\s*(?:enrollment|recruitment).*?(\d+)'
        ]
        
        for pattern in size_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return 'not_specified'
    
    def _extract_intervention(self, arm_text: str) -> str:
        """Extract intervention from treatment arm description."""
        # Look for drug names (capitalize first letter of words)
        drug_pattern = r'[A-Z][a-z]+(?:-[A-Z][a-z]+)*(?:\s+\d+)?'
        matches = re.findall(drug_pattern, arm_text)
        
        if matches:
            return ', '.join(matches[:3])  # Limit to first 3 matches
        
        return 'not_specified'
    
    def _extract_dose(self, text: str) -> str:
        """Extract dose information."""
        dose_match = re.search(r'(\d+(?:\.\d+)?)\s*(mg|g|mcg|units?)', text, re.IGNORECASE)
        if dose_match:
            return f"{dose_match.group(1)} {dose_match.group(2)}"
        return 'not_specified'
    
    def _extract_frequency(self, text: str) -> str:
        """Extract dosing frequency."""
        freq_patterns = {
            'once daily': r'once\s*daily|QD|q24h',
            'twice daily': r'twice\s*daily|BID|q12h',
            'three times daily': r'three\s*times\s*daily|TID|q8h',
            'four times daily': r'four\s*times\s*daily|QID|q6h',
            'weekly': r'weekly|once\s*per\s*week',
            'monthly': r'monthly|once\s*per\s*month'
        }
        
        for freq, pattern in freq_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return freq
        
        return 'not_specified'
    
    def _extract_frequency_from_context(self, context: str) -> str:
        """Extract frequency from context text."""
        return self._extract_frequency(context)
    
    def _extract_route(self, context: str) -> str:
        """Extract route of administration."""
        routes = {
            'oral': r'oral|PO|by\s*mouth',
            'intravenous': r'intravenous|IV|i\.v\.',
            'subcutaneous': r'subcutaneous|SC|s\.c\.',
            'intramuscular': r'intramuscular|IM|i\.m\.',
            'topical': r'topical|applied\s*to\s*skin'
        }
        
        for route, pattern in routes.items():
            if re.search(pattern, context, re.IGNORECASE):
                return route
        
        return 'not_specified'
    
    def _extract_duration_from_context(self, context: str) -> str:
        """Extract treatment duration from context."""
        duration_match = re.search(
            r'(?:for|duration|period).*?(\d+)\s*(?:days?|weeks?|months?|years?)',
            context,
            re.IGNORECASE
        )
        if duration_match:
            return duration_match.group(1)
        return 'not_specified'
    
    def _classify_population(self, pop_name: str) -> str:
        """Classify population type."""
        name_lower = pop_name.lower()
        
        if 'safety' in name_lower:
            return 'safety'
        elif 'efficacy' in name_lower:
            return 'efficacy'
        elif 'itt' in name_lower or 'intent' in name_lower:
            return 'intent_to_treat'
        elif 'protocol' in name_lower:
            return 'per_protocol'
        elif 'evaluable' in name_lower:
            return 'evaluable'
        else:
            return 'analysis'
    
    def _calculate_completeness(self, protocol_data: ProtocolData) -> float:
        """Calculate completeness score for protocol extraction."""
        scores = []
        
        # Core protocol elements (weighted scoring)
        weights = {
            'inclusion_criteria': 0.20,
            'exclusion_criteria': 0.20,
            'primary_endpoints': 0.25,
            'study_design': 0.10,
            'treatment_arms': 0.10,
            'secondary_endpoints': 0.05,
            'schedule_of_assessments': 0.05,
            'planned_procedures': 0.05
        }
        
        # Inclusion criteria
        inclusion_score = min(1.0, len(protocol_data.inclusion_criteria) / 5.0)
        scores.append(inclusion_score * weights['inclusion_criteria'])
        
        # Exclusion criteria
        exclusion_score = min(1.0, len(protocol_data.exclusion_criteria) / 5.0)
        scores.append(exclusion_score * weights['exclusion_criteria'])
        
        # Primary endpoints
        primary_score = min(1.0, len(protocol_data.primary_endpoints) / 1.0)
        scores.append(primary_score * weights['primary_endpoints'])
        
        # Study design
        design_fields = ['design_type', 'phase', 'randomization', 'blinding']
        design_score = sum(1 for field in design_fields 
                          if protocol_data.study_design.get(field) != 'unknown') / len(design_fields)
        scores.append(design_score * weights['study_design'])
        
        # Treatment arms
        arms_score = min(1.0, len(protocol_data.treatment_arms) / 2.0)
        scores.append(arms_score * weights['treatment_arms'])
        
        # Secondary endpoints
        secondary_score = min(1.0, len(protocol_data.secondary_endpoints) / 2.0)
        scores.append(secondary_score * weights['secondary_endpoints'])
        
        # Schedule of assessments
        schedule_score = min(1.0, len(protocol_data.schedule_of_assessments) / 3.0)
        scores.append(schedule_score * weights['schedule_of_assessments'])
        
        # Planned procedures
        procedures_score = min(1.0, len(protocol_data.planned_procedures) / 5.0)
        scores.append(procedures_score * weights['planned_procedures'])
        
        return sum(scores)
    
    def _get_sections_found(self, protocol_data: ProtocolData) -> List[str]:
        """Get list of sections that were successfully extracted."""
        sections = []
        
        section_checks = [
            ('protocol_summary', lambda: protocol_data.protocol_summary.get('title') != 'unknown'),
            ('estimands', lambda: len(protocol_data.estimands) > 0),
            ('inclusion_criteria', lambda: len(protocol_data.inclusion_criteria) > 0),
            ('exclusion_criteria', lambda: len(protocol_data.exclusion_criteria) > 0),
            ('primary_endpoints', lambda: len(protocol_data.primary_endpoints) > 0),
            ('secondary_endpoints', lambda: len(protocol_data.secondary_endpoints) > 0),
            ('study_design', lambda: protocol_data.study_design.get('design_type') != 'unknown'),
            ('treatment_arms', lambda: len(protocol_data.treatment_arms) > 0),
            ('schedule_of_assessments', lambda: len(protocol_data.schedule_of_assessments) > 0),
            ('planned_procedures', lambda: len(protocol_data.planned_procedures) > 0),
            ('dosing_regimens', lambda: len(protocol_data.dosing_regimens) > 0),
            ('discontinuation', lambda: len(protocol_data.discontinuation) > 0),
            ('adverse_events', lambda: protocol_data.adverse_events.get('reporting_procedures') != 'unknown'),
            ('statistical_analyses', lambda: len(protocol_data.statistical_analyses) > 0),
            ('pharmacokinetics', lambda: len(protocol_data.pharmacokinetics) > 0)
        ]
        
        for section_name, check_func in section_checks:
            if check_func():
                sections.append(section_name)
        
        return sections
