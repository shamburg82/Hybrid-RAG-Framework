# structured_extractors/sap_extractor.py
"""
SAP Extractor - Structured extraction from Statistical Analysis Plan documents

Extracts key structured information including:
- Analysis populations and their definitions
- Statistical methods and procedures
- Analysis specifications by endpoint
- Planned analyses and their details
"""

import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from llama_index.core import Document
from .base_extractor import BaseExtractor, StructuredData, ExtractionResult


@dataclass
class SAPData(StructuredData):
    """Structured data extracted from SAP documents."""
    analysis_populations: List[Dict[str, Any]]
    statistical_methods: List[Dict[str, Any]]
    analysis_specifications: List[Dict[str, Any]]
    endpoints_analysis: List[Dict[str, Any]]
    planned_analyses: List[Dict[str, Any]]
    interim_analyses: List[Dict[str, Any]]
    multiplicity_adjustments: List[Dict[str, Any]]
    missing_data_handling: Dict[str, Any]


class SAPExtractor(BaseExtractor):
    """Extracts structured information from Statistical Analysis Plan documents."""
    
    def __init__(self):
        super().__init__()
        self.document_type = "sap"
        
        # Patterns for identifying SAP sections
        self.section_patterns = {
            'analysis_populations': [
                r'analysis\s+(?:populations|sets)',
                r'study\s+populations',
                r'analysis\s+population\s+definitions',
                r'efficacy\s+populations',
                r'safety\s+population'
            ],
            'statistical_methods': [
                r'statistical\s+methods?',
                r'statistical\s+procedures?',
                r'analytical\s+methods?',
                r'statistical\s+approach'
            ],
            'analysis_specifications': [
                r'analysis\s+specifications?',
                r'planned\s+analyses',
                r'statistical\s+analysis\s+plan',
                r'analysis\s+details'
            ],
            'endpoints_analysis': [
                r'(?:primary|secondary)\s+endpoint\s+analysis',
                r'efficacy\s+analysis',
                r'endpoint\s+analyses'
            ],
            'interim_analyses': [
                r'interim\s+analysis',
                r'interim\s+analyses',
                r'data\s+monitoring'
            ],
            'missing_data': [
                r'missing\s+data',
                r'data\s+imputation',
                r'handling\s+of\s+missing'
            ]
        }
        
        # Statistical method patterns
        self.statistical_patterns = {
            'descriptive': r'descriptive\s+statistics?',
            'chi_square': r'chi.?square|χ²\s*test',
            'fishers_exact': r'Fisher\'?s?\s+exact\s+test',
            't_test': r't.?test|Student\'?s?\s+t.?test',
            'wilcoxon': r'Wilcoxon|Mann.?Whitney',
            'anova': r'ANOVA|analysis\s+of\s+variance',
            'ancova': r'ANCOVA|analysis\s+of\s+covariance',
            'logistic_regression': r'logistic\s+regression',
            'cox_regression': r'Cox\s+regression|proportional\s+hazards',
            'kaplan_meier': r'Kaplan.?Meier|survival\s+analysis',
            'log_rank': r'log.?rank\s+test'
        }
        
        # Population definition patterns
        self.population_patterns = {
            'itt': r'(?:ITT|intention.?to.?treat|intent.?to.?treat)',
            'pp': r'(?:PP|per.?protocol)',
            'safety': r'safety\s+(?:population|set)',
            'efficacy': r'efficacy\s+(?:population|set)',
            'evaluable': r'evaluable\s+(?:population|set)',
            'modified_itt': r'(?:mITT|modified\s+ITT|modified\s+intention.?to.?treat)',
            'fas': r'(?:FAS|full\s+analysis\s+set)'
        }
    
    def extract(self, document: Document) -> ExtractionResult:
        """Extract structured data from SAP document."""
        try:
            text = document.text
            
            # Extract each type of information
            analysis_populations = self._extract_analysis_populations(text)
            statistical_methods = self._extract_statistical_methods(text)
            analysis_specifications = self._extract_analysis_specifications(text)
            endpoints_analysis = self._extract_endpoints_analysis(text)
            planned_analyses = self._extract_planned_analyses(text)
            interim_analyses = self._extract_interim_analyses(text)
            multiplicity_adjustments = self._extract_multiplicity_adjustments(text)
            missing_data_handling = self._extract_missing_data_handling(text)
            
            # Create structured data object
            sap_data = SAPData(
                document_id=document.metadata.get('file_id', 'unknown'),
                document_type='sap',
                extraction_timestamp=self._get_timestamp(),
                confidence_score=self._calculate_confidence([
                    analysis_populations, statistical_methods, analysis_specifications,
                    endpoints_analysis, planned_analyses
                ]),
                analysis_populations=analysis_populations,
                statistical_methods=statistical_methods,
                analysis_specifications=analysis_specifications,
                endpoints_analysis=endpoints_analysis,
                planned_analyses=planned_analyses,
                interim_analyses=interim_analyses,
                multiplicity_adjustments=multiplicity_adjustments,
                missing_data_handling=missing_data_handling
            )
            
            # Calculate completeness
            completeness = self._calculate_completeness(sap_data)
            
            return ExtractionResult(
                success=True,
                data=sap_data,
                completeness_score=completeness,
                extraction_metadata={
                    'sections_found': self._get_sections_found(sap_data),
                    'total_populations': len(analysis_populations),
                    'total_methods': len(statistical_methods),
                    'total_analyses': len(planned_analyses)
                }
            )
            
        except Exception as e:
            return ExtractionResult(
                success=False,
                error=str(e),
                data=None,
                completeness_score=0.0
            )
    
    def _extract_analysis_populations(self, text: str) -> List[Dict[str, Any]]:
        """Extract analysis population definitions."""
        populations = []
        
        # Find the analysis populations section
        section_text = self._find_section(text, 'analysis_populations')
        if not section_text:
            section_text = text  # Search whole document if section not found
        
        # Extract populations using patterns
        for pop_type, pattern in self.population_patterns.items():
            matches = list(re.finditer(pattern, section_text, re.IGNORECASE))
            
            for match in matches:
                # Extract context around the population mention
                start = max(0, match.start() - 300)
                end = min(len(section_text), match.end() + 500)
                context = section_text[start:end]
                
                # Extract definition from context
                definition = self._extract_population_definition(context, pop_type)
                
                population = {
                    'name': self._standardize_population_name(match.group()),
                    'type': pop_type,
                    'definition': definition,
                    'context': self._clean_text(context),
                    'criteria': self._extract_population_criteria(context)
                }
                
                populations.append(population)
        
        # Remove duplicates based on type
        seen_types = set()
        unique_populations = []
        for pop in populations:
            if pop['type'] not in seen_types:
                unique_populations.append(pop)
                seen_types.add(pop['type'])
        
        return unique_populations
    
    def _extract_statistical_methods(self, text: str) -> List[Dict[str, Any]]:
        """Extract statistical methods and procedures."""
        methods = []
        
        # Find statistical methods section
        section_text = self._find_section(text, 'statistical_methods')
        if not section_text:
            section_text = text
        
        # Extract methods using patterns
        for method_name, pattern in self.statistical_patterns.items():
            matches = list(re.finditer(pattern, section_text, re.IGNORECASE))
            
            for match in matches:
                # Extract context around the method mention
                start = max(0, match.start() - 200)
                end = min(len(section_text), match.end() + 400)
                context = section_text[start:end]
                
                method = {
                    'name': method_name.replace('_', ' ').title(),
                    'full_name': match.group(),
                    'description': self._clean_text(context),
                    'application': self._extract_method_application(context),
                    'assumptions': self._extract_method_assumptions(context),
                    'significance_level': self._extract_significance_level(context)
                }
                
                methods.append(method)
        
        # Remove duplicates
        unique_methods = []
        seen_names = set()
        for method in methods:
            if method['name'] not in seen_names:
                unique_methods.append(method)
                seen_names.add(method['name'])
        
        return unique_methods
    
    def _extract_analysis_specifications(self, text: str) -> List[Dict[str, Any]]:
        """Extract detailed analysis specifications."""
        specifications = []
        
        # Find analysis specifications section
        section_text = self._find_section(text, 'analysis_specifications')
        if not section_text:
            # Look for numbered analysis sections
            section_text = self._find_numbered_analyses(text)
        
        if not section_text:
            return specifications
        
        # Extract specifications by parsing structured text
        spec_patterns = [
            r'(\d+\.?\d*\.?\s+)(.*?)(?=\n\s*\d+\.|\Z)',  # Numbered sections
            r'(Analysis\s+\d+:?\s*)(.*?)(?=\nAnalysis\s+\d+|\Z)',  # Analysis labels
            r'(Table\s+\d+\.?\d*\.?\s+)(.*?)(?=\nTable\s+\d+|\Z)'  # Table references
        ]
        
        for pattern in spec_patterns:
            matches = re.findall(pattern, section_text, re.DOTALL | re.IGNORECASE)
            
            for i, (label, content) in enumerate(matches):
                if len(content.strip()) < 50:  # Skip very short content
                    continue
                
                spec = {
                    'id': f"analysis_{i+1}",
                    'label': label.strip(),
                    'description': self._clean_text(content),
                    'endpoint': self._extract_analysis_endpoint(content),
                    'population': self._extract_analysis_population(content),
                    'method': self._extract_analysis_method(content),
                    'variables': self._extract_analysis_variables(content),
                    'comparisons': self._extract_analysis_comparisons(content)
                }
                
                specifications.append(spec)
        
        return specifications
    
    def _extract_endpoints_analysis(self, text: str) -> List[Dict[str, Any]]:
        """Extract endpoint-specific analysis details."""
        endpoint_analyses = []
        
        # Look for primary and secondary endpoint analysis sections
        endpoint_patterns = [
            r'(primary\s+endpoint\s+analysis)\s*:?\s*(.*?)(?=secondary\s+endpoint|interim\s+analysis|\n\s*\d+\.|\Z)',
            r'(secondary\s+endpoint\s+analysis)\s*:?\s*(.*?)(?=interim\s+analysis|\n\s*\d+\.|\Z)',
            r'(efficacy\s+analysis)\s*:?\s*(.*?)(?=safety\s+analysis|\n\s*\d+\.|\Z)'
        ]
        
        for pattern in endpoint_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            
            for endpoint_type, content in matches:
                if len(content.strip()) < 50:
                    continue
                
                analysis = {
                    'endpoint_type': endpoint_type.lower().replace(' ', '_'),
                    'description': self._clean_text(content),
                    'statistical_test': self._extract_statistical_test(content),
                    'hypothesis': self._extract_hypothesis(content),
                    'alpha_level': self._extract_alpha_level(content),
                    'power': self._extract_power(content),
                    'effect_size': self._extract_effect_size(content)
                }
                
                endpoint_analyses.append(analysis)
        
        return endpoint_analyses
    
    def _extract_planned_analyses(self, text: str) -> List[Dict[str, Any]]:
        """Extract all planned analyses from the SAP."""
        analyses = []
        
        # Look for analysis tables or numbered lists
        analysis_patterns = [
            r'Table\s+(\d+\.?\d*)\s*[:\-]?\s*(.*?)(?=Table\s+\d+|\n\n|\Z)',
            r'Analysis\s+(\d+\.?\d*)\s*[:\-]?\s*(.*?)(?=Analysis\s+\d+|\n\n|\Z)',
            r'(\d+\.?\d*\.?\s+)(.*?)(?=\n\s*\d+\.|\Z)'
        ]
        
        for pattern in analysis_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            
            for i, match in enumerate(matches):
                if len(match) == 2:
                    analysis_id, content = match
                else:
                    analysis_id = str(i + 1)
                    content = match[0] if isinstance(match, tuple) else str(match)
                
                if len(content.strip()) < 30:
                    continue
                
                analysis = {
                    'analysis_id': analysis_id.strip(),
                    'description': self._clean_text(content),
                    'analysis_type': self._classify_analysis_type(content),
                    'endpoint': self._extract_analysis_endpoint(content),
                    'population': self._extract_analysis_population(content),
                    'method': self._extract_analysis_method(content),
                    'timing': self._extract_analysis_timing(content)
                }
                
                analyses.append(analysis)
        
        return analyses
    
    def _extract_interim_analyses(self, text: str) -> List[Dict[str, Any]]:
        """Extract interim analysis specifications."""
        interim_analyses = []
        
        # Find interim analysis section
        section_text = self._find_section(text, 'interim_analyses')
        if not section_text:
            return interim_analyses
        
        # Extract interim analysis details
        interim_patterns = [
            r'interim\s+analysis\s+(\d+)\s*:?\s*(.*?)(?=interim\s+analysis\s+\d+|\Z)',
            r'(\d+)(?:st|nd|rd|th)\s+interim\s+analysis\s*:?\s*(.*?)(?=\d+(?:st|nd|rd|th)\s+interim|\Z)'
        ]
        
        for pattern in interim_patterns:
            matches = re.findall(pattern, section_text, re.DOTALL | re.IGNORECASE)
            
            for analysis_num, content in matches:
                if len(content.strip()) < 30:
                    continue
                
                interim = {
                    'analysis_number': analysis_num,
                    'description': self._clean_text(content),
                    'timing': self._extract_interim_timing(content),
                    'stopping_rules': self._extract_stopping_rules(content),
                    'efficacy_boundary': self._extract_efficacy_boundary(content),
                    'futility_boundary': self._extract_futility_boundary(content)
                }
                
                interim_analyses.append(interim)
        
        return interim_analyses
    
    def _extract_multiplicity_adjustments(self, text: str) -> List[Dict[str, Any]]:
        """Extract multiplicity adjustment procedures."""
        adjustments = []
        
        # Look for multiplicity-related sections
        multiplicity_patterns = [
            r'multiplicity\s+adjustment',
            r'multiple\s+comparison',
            r'Bonferroni',
            r'Holm',
            r'Benjamini.?Hochberg',
            r'false\s+discovery\s+rate',
            r'family.?wise\s+error\s+rate'
        ]
        
        for pattern in multiplicity_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            
            for match in matches:
                # Extract context
                start = max(0, match.start() - 200)
                end = min(len(text), match.end() + 300)
                context = text[start:end]
                
                adjustment = {
                    'method': match.group(),
                    'description': self._clean_text(context),
                    'alpha_allocation': self._extract_alpha_allocation(context),
                    'procedure_details': self._extract_procedure_details(context)
                }
                
                adjustments.append(adjustment)
        
        return adjustments
    
    def _extract_missing_data_handling(self, text: str) -> Dict[str, Any]:
        """Extract missing data handling procedures."""
        # Find missing data section
        section_text = self._find_section(text, 'missing_data')
        if not section_text:
            section_text = text
        
        missing_data = {
            'imputation_method': 'not_specified',
            'sensitivity_analyses': [],
            'assumptions': 'not_specified',
            'procedures': []
        }
        
        # Extract imputation methods
        imputation_patterns = {
            'LOCF': r'LOCF|last\s+observation\s+carried\s+forward',
            'MI': r'multiple\s+imputation',
            'MMRM': r'MMRM|mixed\s+model\s+repeated\s+measures',
            'complete_case': r'complete\s+case\s+analysis',
            'listwise_deletion': r'listwise\s+deletion'
        }
        
        for method, pattern in imputation_patterns.items():
            if re.search(pattern, section_text, re.IGNORECASE):
                missing_data['imputation_method'] = method
                break
        
        # Extract sensitivity analyses
        sensitivity_match = re.search(
            r'sensitivity\s+analys[ie]s\s*:?\s*(.*?)(?=\n\s*\d+\.|\n\n|\Z)',
            section_text,
            re.DOTALL | re.IGNORECASE
        )
        
        if sensitivity_match:
            missing_data['sensitivity_analyses'] = [
                self._clean_text(sensitivity_match.group(1))
            ]
        
        return missing_data
    
    # Helper methods
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
    
    def _find_numbered_analyses(self, text: str) -> str:
        """Find numbered analysis sections."""
        # Look for sections with multiple numbered items
        numbered_pattern = r'(\d+\.?\d*\.?\s+.*?)(?=\n\s*\d+\.|\Z)'
        matches = re.findall(numbered_pattern, text, re.DOTALL)
        
        if len(matches) >= 3:  # At least 3 numbered items
            return '\n'.join(matches)
        
        return ""
    
    def _extract_population_definition(self, context: str, pop_type: str) -> str:
        """Extract population definition from context."""
        # Look for definition patterns
        def_patterns = [
            r'defined\s+as\s+(.*?)(?=\.|;|\n)',
            r'includes?\s+(.*?)(?=\.|;|\n)',
            r'consists?\s+of\s+(.*?)(?=\.|;|\n)',
            r'comprises?\s+(.*?)(?=\.|;|\n)'
        ]
        
        for pattern in def_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                return self._clean_text(match.group(1))
        
        return 'not_specified'
    
    def _standardize_population_name(self, raw_name: str) -> str:
        """Standardize population names."""
        name_mapping = {
            'itt': 'Intent-to-Treat (ITT)',
            'pp': 'Per-Protocol (PP)',
            'safety': 'Safety Population',
            'efficacy': 'Efficacy Population',
            'evaluable': 'Evaluable Population',
            'modified_itt': 'Modified Intent-to-Treat (mITT)',
            'fas': 'Full Analysis Set (FAS)'
        }
        
        for key, standard_name in name_mapping.items():
            if key in raw_name.lower():
                return standard_name
        
        return raw_name
    
    def _extract_population_criteria(self, context: str) -> List[str]:
        """Extract population inclusion criteria."""
        criteria = []
        
        # Look for bulleted or numbered criteria
        criteria_patterns = [
            r'(?:•|\*|\-|\d+\.)\s+(.*?)(?=\n(?:•|\*|\-|\d+\.)|$)',
            r'includes?\s+(.*?)(?=\.|;|\n)',
            r'must\s+(.*?)(?=\.|;|\n)'
        ]
        
        for pattern in criteria_patterns:
            matches = re.findall(pattern, context, re.IGNORECASE)
            for match in matches:
                clean_criterion = self._clean_text(match)
                if len(clean_criterion) > 10:
                    criteria.append(clean_criterion)
        
        return criteria[:5]  # Limit to first 5 criteria
    
    def _extract_method_application(self, context: str) -> str:
        """Extract where/how the statistical method is applied."""
        app_patterns = [
            r'(?:used|applied|employed)\s+(?:for|to)\s+(.*?)(?=\.|;|\n)',
            r'will\s+be\s+used\s+to\s+(.*?)(?=\.|;|\n)',
            r'analysis\s+of\s+(.*?)(?=\.|;|\n)'
        ]
        
        for pattern in app_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                return self._clean_text(match.group(1))
        
        return 'not_specified'
    
    def _extract_method_assumptions(self, context: str) -> List[str]:
        """Extract statistical method assumptions."""
        assumptions = []
        
        assumption_keywords = [
            'normal', 'normality', 'gaussian', 'independence', 'homoscedasticity',
            'proportional hazards', 'linearity', 'homogeneity'
        ]
        
        for keyword in assumption_keywords:
            if keyword in context.lower():
                assumptions.append(keyword)
        
        return assumptions
    
    def _extract_significance_level(self, context: str) -> str:
        """Extract significance level (alpha)."""
        alpha_patterns = [
            r'α\s*=\s*(0?\.\d+)',
            r'alpha\s*=\s*(0?\.\d+)',
            r'significance\s+level\s+of\s+(0?\.\d+)',
            r'p\s*<\s*(0?\.\d+)'
        ]
        
        for pattern in alpha_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return '0.05'  # Default
    
    def _extract_analysis_endpoint(self, content: str) -> str:
        """Extract the endpoint being analyzed."""
        endpoint_patterns = [
            r'(?:primary|secondary)\s+endpoint',
            r'(?:efficacy|safety)\s+endpoint',
            r'overall\s+survival',
            r'progression.?free\s+survival',
            r'response\s+rate',
            r'adverse\s+events?'
        ]
        
        for pattern in endpoint_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group()
        
        return 'not_specified'
    
    def _extract_analysis_population(self, content: str) -> str:
        """Extract the analysis population."""
        for pop_type, pattern in self.population_patterns.items():
            if re.search(pattern, content, re.IGNORECASE):
                return pop_type
        
        return 'not_specified'
    
    def _extract_analysis_method(self, content: str) -> str:
        """Extract the statistical method for analysis."""
        for method_name, pattern in self.statistical_patterns.items():
            if re.search(pattern, content, re.IGNORECASE):
                return method_name.replace('_', ' ').title()
        
        return 'not_specified'
    
    def _extract_analysis_variables(self, content: str) -> List[str]:
        """Extract analysis variables."""
        variables = []
        
        var_patterns = [
            r'variable[s]?\s*:?\s*(.*?)(?=\n|\.|;)',
            r'covariate[s]?\s*:?\s*(.*?)(?=\n|\.|;)',
            r'factor[s]?\s*:?\s*(.*?)(?=\n|\.|;)'
        ]
        
        for pattern in var_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                vars_text = match.group(1)
                # Split on common delimiters
                vars_list = re.split(r'[,;]', vars_text)
                variables.extend([v.strip() for v in vars_list if v.strip()])
        
        return variables[:5]  # Limit to first 5
    
    def _extract_analysis_comparisons(self, content: str) -> List[str]:
        """Extract planned comparisons."""
        comparisons = []
        
        comp_patterns = [
            r'(?:versus|vs\.?|compared\s+to)\s+(.*?)(?=\n|\.|;)',
            r'treatment\s+(?:group\s+)?[A-Z]\s+(?:versus|vs\.?)\s+(?:group\s+)?[A-Z]',
            r'active\s+(?:versus|vs\.?)\s+placebo'
        ]
        
        for pattern in comp_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            comparisons.extend([self._clean_text(match) for match in matches])
        
        return comparisons[:3]  # Limit to first 3
    
    def _extract_statistical_test(self, content: str) -> str:
        """Extract specific statistical test."""
        return self._extract_analysis_method(content)
    
    def _extract_hypothesis(self, content: str) -> str:
        """Extract hypothesis being tested."""
        hyp_patterns = [
            r'(?:null|alternative)\s+hypothesis\s*:?\s*(.*?)(?=\n|\.|;)',
            r'H[01]\s*:?\s*(.*?)(?=\n|\.|;)',
            r'test\s+(?:that|whether)\s+(.*?)(?=\n|\.|;)'
        ]
        
        for pattern in hyp_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return self._clean_text(match.group(1))
        
        return 'not_specified'
    
    def _extract_alpha_level(self, content: str) -> str:
        """Extract alpha level for hypothesis test."""
        return self._extract_significance_level(content)
    
    def _extract_power(self, content: str) -> str:
        """Extract statistical power."""
        power_patterns = [
            r'power\s+of\s+(\d+%|\d*\.\d+)',
            r'(\d+%|\d*\.\d+)\s+power',
            r'β\s*=\s*(0?\.\d+)'
        ]
        
        for pattern in power_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return 'not_specified'
    
    def _extract_effect_size(self, content: str) -> str:
        """Extract expected effect size."""
        effect_patterns = [
            r'effect\s+size\s+of\s+(.*?)(?=\n|\.|;)',
            r'difference\s+of\s+(.*?)(?=\n|\.|;)',
            r'hazard\s+ratio\s+of\s+(.*?)(?=\n|\.|;)'
        ]
        
        for pattern in effect_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return self._clean_text(match.group(1))
        
        return 'not_specified'
    
    def _classify_analysis_type(self, content: str) -> str:
        """Classify the type of analysis."""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['primary', 'main', 'principal']):
            return 'primary'
        elif any(word in content_lower for word in ['secondary', 'supplementary']):
            return 'secondary'
        elif any(word in content_lower for word in ['exploratory', 'post-hoc']):
            return 'exploratory'
        elif any(word in content_lower for word in ['sensitivity', 'robustness']):
            return 'sensitivity'
        elif any(word in content_lower for word in ['subgroup', 'subset']):
            return 'subgroup'
        else:
            return 'general'
    
    def _extract_analysis_timing(self, content: str) -> str:
        """Extract when the analysis will be performed."""
        timing_patterns = [
            r'(?:at|after)\s+(\d+\s+(?:weeks?|months?|years?))',
            r'(?:end\s+of\s+(?:study|treatment))',
            r'(?:interim|final)\s+analysis',
            r'(?:baseline|screening|follow.?up)'
        ]
        
        for pattern in timing_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group()
        
        return 'not_specified'
    
    def _extract_interim_timing(self, content: str) -> str:
        """Extract timing for interim analysis."""
        timing_patterns = [
            r'after\s+(\d+)\s+(?:patients?|subjects?|events?)',
            r'at\s+(\d+%|\d+\s+percent)\s+(?:enrollment|events?)',
            r'(\d+)\s+(?:months?|years?)\s+after'
        ]
        
        for pattern in timing_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group()
        
        return 'not_specified'
    
    def _extract_stopping_rules(self, content: str) -> List[str]:
        """Extract stopping rules for interim analysis."""
        rules = []
        
        rule_patterns = [
            r'stop\s+(?:for|if)\s+(.*?)(?=\n|\.|;)',
            r'stopping\s+rule\s*:?\s*(.*?)(?=\n|\.|;)',
            r'futility\s+(?:boundary|threshold)\s*:?\s*(.*?)(?=\n|\.|;)'
        ]
        
        for pattern in rule_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            rules.extend([self._clean_text(match) for match in matches])
        
        return rules
    
    def _extract_efficacy_boundary(self, content: str) -> str:
        """Extract efficacy boundary for interim analysis."""
        boundary_patterns = [
            r'efficacy\s+boundary\s*:?\s*(.*?)(?=\n|\.|;)',
            r'α.?spending\s+function\s*:?\s*(.*?)(?=\n|\.|;)',
            r'O\'?Brien.?Fleming'
        ]
        
        for pattern in boundary_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return self._clean_text(match.group(1) if len(match.groups()) > 0 else match.group())
        
        return 'not_specified'
    
    def _extract_futility_boundary(self, content: str) -> str:
        """Extract futility boundary for interim analysis."""
        futility_patterns = [
            r'futility\s+boundary\s*:?\s*(.*?)(?=\n|\.|;)',
            r'conditional\s+power\s*<\s*(.*?)(?=\n|\.|;)',
            r'β.?spending\s+function\s*:?\s*(.*?)(?=\n|\.|;)'
        ]
        
        for pattern in futility_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return self._clean_text(match.group(1))
        
        return 'not_specified'
    
    def _extract_alpha_allocation(self, context: str) -> str:
        """Extract alpha allocation for multiplicity adjustment."""
        alpha_patterns = [
            r'α\s*=\s*(0?\.\d+)',
            r'alpha\s+allocated\s*:?\s*(.*?)(?=\n|\.|;)',
            r'(\d+%|\d*\.\d+)\s+(?:of\s+)?alpha'
        ]
        
        for pattern in alpha_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return 'not_specified'
    
    def _extract_procedure_details(self, context: str) -> str:
        """Extract procedure details for multiplicity adjustment."""
        return self._clean_text(context)
    
    def _calculate_completeness(self, sap_data: SAPData) -> float:
        """Calculate completeness score for SAP extraction."""
        scores = []
        
        # Analysis populations (weight: 0.25)
        pop_score = min(1.0, len(sap_data.analysis_populations) / 3.0)
        scores.append(pop_score * 0.25)
        
        # Statistical methods (weight: 0.2)
        methods_score = min(1.0, len(sap_data.statistical_methods) / 5.0)
        scores.append(methods_score * 0.2)
        
        # Analysis specifications (weight: 0.25)
        specs_score = min(1.0, len(sap_data.analysis_specifications) / 5.0)
        scores.append(specs_score * 0.25)
        
        # Endpoints analysis (weight: 0.15)
        endpoints_score = min(1.0, len(sap_data.endpoints_analysis) / 2.0)
        scores.append(endpoints_score * 0.15)
        
        # Planned analyses (weight: 0.15)
        planned_score = min(1.0, len(sap_data.planned_analyses) / 10.0)
        scores.append(planned_score * 0.15)
        
        return sum(scores)
    
    def _get_sections_found(self, sap_data: SAPData) -> List[str]:
        """Get list of sections that were successfully extracted."""
        sections = []
        
        if sap_data.analysis_populations:
            sections.append('analysis_populations')
        if sap_data.statistical_methods:
            sections.append('statistical_methods')
        if sap_data.analysis_specifications:
            sections.append('analysis_specifications')
        if sap_data.endpoints_analysis:
            sections.append('endpoints_analysis')
        if sap_data.planned_analyses:
            sections.append('planned_analyses')
        if sap_data.interim_analyses:
            sections.append('interim_analyses')
        if sap_data.multiplicity_adjustments:
            sections.append('multiplicity_adjustments')
        if sap_data.missing_data_handling.get('imputation_method') != 'not_specified':
            sections.append('missing_data_handling')
        
        return sections
