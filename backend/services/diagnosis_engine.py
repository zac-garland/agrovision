"""
Diagnosis engine that synthesizes plant identification, health analysis, and generates
comprehensive diagnosis with treatment recommendations using LLM or rule-based logic.
"""

from typing import Dict, List, Optional
from models.llm_model import get_llm_model


class DiagnosisEngine:
    """Engine for generating plant diagnosis and treatment recommendations."""
    
    def __init__(self):
        """Initialize diagnosis engine."""
        self.llm = get_llm_model()
    
    def synthesize_diagnosis(
        self,
        plant_species: Dict,
        leaf_analysis: Dict,
        use_llm: bool = True
    ) -> Dict:
        """
        Synthesize diagnosis from all available data.
        
        Args:
            plant_species: Plant identification results from PlantNet
            leaf_analysis: Leaf health analysis results
            use_llm: Whether to use LLM (True) or rule-based (False)
            
        Returns:
            Dictionary with:
            - 'final_diagnosis': Condition, severity, reasoning
            - 'treatment_plan': Immediate, week_1, week_2_3, monitoring
            - 'source': 'llm' or 'rule_based'
        """
        if use_llm and self.llm.available:
            return self._llm_diagnosis(plant_species, leaf_analysis)
        else:
            return self._rule_based_diagnosis(plant_species, leaf_analysis)
    
    def _llm_diagnosis(self, plant_species: Dict, leaf_analysis: Dict) -> Dict:
        """Generate diagnosis using LLM."""
        try:
            # Build prompt from all available data
            prompt = self._build_diagnosis_prompt(plant_species, leaf_analysis)
            
            system_prompt = self._get_system_prompt()
            
            # Generate reasoning
            result = self.llm.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=512
            )
            
            if result['success']:
                # Parse LLM response
                return self._parse_llm_response(result['text'], plant_species, leaf_analysis)
            else:
                # Fallback to rule-based if LLM fails
                print(f"⚠️  LLM generation failed: {result.get('error')}")
                print("   Falling back to rule-based recommendations")
                return self._rule_based_diagnosis(plant_species, leaf_analysis)
                
        except Exception as e:
            print(f"❌ Error in LLM diagnosis: {e}")
            return self._rule_based_diagnosis(plant_species, leaf_analysis)
    
    def _rule_based_diagnosis(self, plant_species: Dict, leaf_analysis: Dict) -> Dict:
        """Generate diagnosis using rule-based logic."""
        health_score = leaf_analysis.get('overall_health_score', 1.0)
        has_issues = leaf_analysis.get('has_potential_issues', False)
        
        # Determine condition and severity
        if health_score >= 0.8:
            condition = "Plant appears healthy"
            severity = "none"
            reasoning = f"The {plant_species.get('common_name', 'plant')} appears to be in good health. The leaf analysis shows {health_score*100:.0f}% health score."
        elif health_score >= 0.6:
            condition = "Minor health concerns detected"
            severity = "low"
            reasoning = f"Some signs of stress or minor issues detected in the {plant_species.get('common_name', 'plant')}. Health score: {health_score*100:.0f}%."
        elif health_score >= 0.4:
            condition = "Moderate health issues detected"
            severity = "moderate"
            reasoning = f"Several health issues detected in the {plant_species.get('common_name', 'plant')}. Health score: {health_score*100:.0f}%."
        else:
            condition = "Significant health problems detected"
            severity = "high"
            reasoning = f"Serious health issues detected in the {plant_species.get('common_name', 'plant')}. Health score: {health_score*100:.0f}%."
        
        # Generate treatment plan based on severity
        treatment_plan = self._generate_treatment_plan(severity, health_score, leaf_analysis)
        
        return {
            'final_diagnosis': {
                'condition': condition,
                'confidence': round(1.0 - health_score, 3) if health_score < 1.0 else 0.0,
                'severity': severity,
                'reasoning': reasoning
            },
            'treatment_plan': treatment_plan,
            'source': 'rule_based'
        }
    
    def _build_diagnosis_prompt(self, plant_species: Dict, leaf_analysis: Dict) -> str:
        """Build prompt for LLM diagnosis."""
        plant_name = plant_species.get('common_name', 'Unknown plant')
        scientific_name = plant_species.get('species_name', 'Unknown')
        health_score = leaf_analysis.get('overall_health_score', 1.0)
        num_leaves = leaf_analysis.get('num_leaves_detected', 1)
        has_issues = leaf_analysis.get('has_potential_issues', False)
        
        # Use common name as the primary name, fallback to extracting from scientific name if needed
        if plant_name == 'Unknown plant' or not plant_name or plant_name == scientific_name:
            # Try to extract a simpler name from scientific name (e.g., "Fittonia albivenis" -> "Fittonia")
            if scientific_name and scientific_name != 'Unknown':
                parts = scientific_name.split()
                if parts:
                    plant_name = parts[0]  # Use first word of scientific name as fallback
        
        # Get lesion-focused metrics for emphasis
        individual_leaves = leaf_analysis.get('individual_leaves', [])
        avg_lesion_pct = 0.0
        total_lesion_regions = 0
        if individual_leaves:
            avg_lesion_pct = sum(leaf.get('lesion_percentage', 0) for leaf in individual_leaves) / len(individual_leaves)
            total_lesion_regions = sum(leaf.get('num_lesion_regions', 0) for leaf in individual_leaves)
        
        prompt = f"""Analyze the following plant health data and provide a comprehensive diagnosis:

PLANT INFORMATION:
- Common Name: {plant_name}
- Scientific Name (for reference only): {scientific_name}
- Identification Confidence: {plant_species.get('confidence', 0)*100:.1f}%

IMPORTANT: Always refer to this plant by its common name "{plant_name}" throughout your response. Do not use the scientific name when addressing the user.

HEALTH ANALYSIS:
- Overall Health Score: {health_score:.2f} (1.0 = perfect health, primarily based on lesion detection)
- Leaves Analyzed: {num_leaves}
- Potential Issues Detected: {'Yes' if has_issues else 'No'}

PRIMARY HEALTH INDICATORS (Most Important):
- Average Lesion Coverage: {avg_lesion_pct:.1f}% of leaf area shows potential lesions/damage
- Total Lesion Regions Detected: {total_lesion_regions}
- Note: Lesion detection is the primary health indicator. Green color percentage is less relevant as many healthy plants have non-green foliage (purple, red, variegated, etc.).

LEAF DETAILS:"""
        
        if individual_leaves:
            for i, leaf in enumerate(individual_leaves[:3], 1):
                lesion_pct = leaf.get('lesion_percentage', 0)
                lesion_regions = leaf.get('num_lesion_regions', 0)
                green_pct = leaf.get('green_percentage', 0)
                prompt += f"""
  Leaf {i}:
  - Lesion Coverage: {lesion_pct:.1f}% (PRIMARY CONCERN)
  - Lesion Regions: {lesion_regions} distinct areas of concern
  - Green Percentage: {green_pct:.1f}% (secondary indicator - note: many healthy plants aren't green)
  - Health Score: {leaf.get('health_score', 0):.2f}"""
        
        prompt += f"""

Based on this information, provide:
1. A clear diagnosis of this {plant_name}'s condition (use the common name "{plant_name}")
   - Focus primarily on the lesion coverage and lesion regions detected
   - Lesion percentage is the main health indicator
   - Do not rely heavily on green color as many healthy plants have non-green leaves
2. An assessment of severity (none/low/moderate/high)
   - Base severity primarily on lesion percentage: <3% = none/low, 3-10% = moderate, >10% = high
3. A brief explanation of your reasoning
   - Explain what the lesion coverage indicates
   - Mention specific concerns based on lesion regions detected
4. Practical treatment recommendations organized by:
   - Immediate actions (within 24 hours)
   - Week 1 care steps
   - Weeks 2-3 care steps
   - Ongoing monitoring advice

IMPORTANT: 
- Always use the common name "{plant_name}" when referring to the plant in your response
- Focus on lesion detection as the primary health indicator
- Do not assume that low green percentage indicates poor health (many healthy plants are purple, red, variegated, etc.)
- Format your response as clear, actionable advice for a home gardener."""
        
        return prompt
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for LLM."""
        return """You are an expert plant pathologist and horticulturalist specializing in plant health diagnosis and care. 
You provide clear, practical, and actionable advice for home gardeners. 
Your diagnoses are based on scientific understanding of plant biology, diseases, and stress factors.
You communicate in a helpful, reassuring tone and provide specific, actionable steps.

CRITICAL DIAGNOSIS GUIDELINES:
- Lesion detection (spots, browning, yellowing, damage) is the PRIMARY indicator of plant health
- Do NOT rely on green color percentage as a health indicator - many perfectly healthy plants have purple, red, variegated, or non-green foliage
- Focus your analysis on lesion coverage percentage and the number of lesion regions detected
- Lesion coverage above 3% typically indicates health concerns that need attention
- Green percentage should only be used as a secondary, minor indicator for green-leaved plants

IMPORTANT: Always use common plant names (e.g., "Fittonia", "Snake Plant", "Pothos") when addressing users. 
Never use scientific names (e.g., "Fittonia albivenis", "Dracaena trifasciata") in your responses to home gardeners. 
Use simple, everyday language that non-experts can understand."""

    def _parse_llm_response(self, text: str, plant_species: Dict, leaf_analysis: Dict) -> Dict:
        """Parse LLM response into structured format."""
        # Extract key information from LLM response
        # For now, use the text as reasoning and extract structured parts
        
        health_score = leaf_analysis.get('overall_health_score', 1.0)
        
        # Replace scientific names with common names in the response
        text = self._replace_scientific_names(text, plant_species)
        
        # Simple extraction - can be improved with better parsing
        lines = text.split('\n')
        
        # Try to extract condition
        condition = "Plant health assessment"
        severity = "moderate"
        
        for line in lines:
            if 'diagnosis' in line.lower() or 'condition' in line.lower():
                condition = line.strip().lstrip('-').lstrip('*').strip()
                if len(condition) > 100:
                    condition = condition[:100] + "..."
            if any(s in line.lower() for s in ['severe', 'serious', 'critical']):
                severity = "high"
            elif any(s in line.lower() for s in ['minor', 'slight', 'low']):
                severity = "low"
            elif 'healthy' in line.lower():
                severity = "none"
        
        # Extract treatment plan sections
        treatment_plan = self._extract_treatment_plan(text)
        
        return {
            'final_diagnosis': {
                'condition': condition,
                'confidence': round(1.0 - health_score, 3) if health_score < 1.0 else 0.0,
                'severity': severity,
                'reasoning': text[:500]  # First 500 chars as reasoning
            },
            'treatment_plan': treatment_plan,
            'source': 'llm'
        }
    
    def _replace_scientific_names(self, text: str, plant_species: Dict) -> str:
        """Replace scientific names in text with common names."""
        import re
        
        common_name = plant_species.get('common_name', '')
        scientific_name = plant_species.get('species_name', '')
        
        # If we have a scientific name, always try to replace it
        if scientific_name and scientific_name != 'Unknown':
            # Replace full scientific name (case-insensitive)
            text = re.sub(re.escape(scientific_name), common_name if common_name else 'this plant', text, flags=re.IGNORECASE)
            
            # Extract genus (first word) from scientific name
            genus = scientific_name.split()[0] if scientific_name else ''
            
            # Replace standalone genus references (word boundaries)
            if genus and len(genus) > 2:
                # Only replace if genus is different from common name (to avoid replacing already correct names)
                if not common_name or genus.lower() != common_name.lower():
                    # Replace "genus" when it appears as a standalone word
                    pattern = r'\b' + re.escape(genus) + r'\b'
                    replacement = common_name if common_name and common_name != scientific_name else 'this plant'
                    text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Also handle common genus names that might need better common names
        # This is a simple lookup for well-known cases
        genus_common_names = {
            'Fittonia': 'Nerve Plant',
            'Epipremnum': 'Pothos',
            'Monstera': 'Monstera',
            'Sansevieria': 'Snake Plant',
            'Dracaena': 'Dracaena',
            'Philodendron': 'Philodendron',
        }
        
        # If common_name is just a genus, try to improve it
        if common_name and common_name in genus_common_names:
            better_name = genus_common_names[common_name]
            # Replace the genus with better name only if they're different
            if better_name.lower() != common_name.lower():
                pattern = r'\b' + re.escape(common_name) + r'\b'
                text = re.sub(pattern, better_name, text, flags=re.IGNORECASE)
                common_name = better_name  # Update for further replacements
        
        return text
    
    def _extract_treatment_plan(self, text: str) -> Dict:
        """Extract treatment plan from LLM response."""
        lines = text.split('\n')
        
        immediate = []
        week_1 = []
        week_2_3 = []
        monitoring = ""
        
        current_section = None
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Detect section headers
            if 'immediate' in line_lower or 'within 24' in line_lower:
                current_section = 'immediate'
                continue
            elif 'week 1' in line_lower or 'first week' in line_lower:
                current_section = 'week_1'
                continue
            elif 'week 2' in line_lower or 'week 3' in line_lower or 'weeks 2-3' in line_lower:
                current_section = 'week_2_3'
                continue
            elif 'monitor' in line_lower or 'ongoing' in line_lower:
                current_section = 'monitoring'
                continue
            
            # Add to current section
            if current_section and line.strip():
                clean_line = line.strip().lstrip('-').lstrip('*').lstrip('1.').lstrip('2.').lstrip('3.').strip()
                if clean_line:
                    if current_section == 'immediate':
                        immediate.append(clean_line)
                    elif current_section == 'week_1':
                        week_1.append(clean_line)
                    elif current_section == 'week_2_3':
                        week_2_3.append(clean_line)
                    elif current_section == 'monitoring':
                        if monitoring:
                            monitoring += " "
                        monitoring += clean_line
        
        # Fallback: if no sections found, distribute recommendations
        if not any([immediate, week_1, week_2_3]):
            all_lines = [l.strip().lstrip('-').lstrip('*').strip() for l in lines if l.strip() and not l.strip()[0].isdigit()]
            all_lines = [l for l in all_lines if len(l) > 10]  # Filter out headers
            
            if all_lines:
                immediate = all_lines[:2] if len(all_lines) >= 2 else all_lines[:1]
                week_1 = all_lines[2:4] if len(all_lines) >= 4 else all_lines[1:3] if len(all_lines) >= 3 else []
                week_2_3 = all_lines[4:] if len(all_lines) > 4 else []
        
        return {
            'immediate': immediate[:3],  # Max 3 items
            'week_1': week_1[:3],
            'week_2_3': week_2_3[:3],
            'monitoring': monitoring[:200] if monitoring else "Monitor the plant daily and adjust care as needed."
        }
    
    def _generate_treatment_plan(self, severity: str, health_score: float, leaf_analysis: Dict) -> Dict:
        """Generate rule-based treatment plan."""
        has_issues = leaf_analysis.get('has_potential_issues', False)
        lesion_pct = leaf_analysis.get('overall_health_score', 1.0)
        
        immediate = []
        week_1 = []
        week_2_3 = []
        monitoring = ""
        
        if severity == "none" or health_score >= 0.8:
            immediate = []
            week_1 = ["Continue current care routine", "Monitor for any changes"]
            week_2_3 = ["Maintain regular watering schedule"]
            monitoring = "Check leaves weekly for any new spots or changes."
        elif severity == "low":
            immediate = ["Inspect all leaves carefully for signs of disease or pests"]
            week_1 = [
                "Remove any visibly damaged or diseased leaves",
                "Ensure proper spacing for good air circulation",
                "Adjust watering to avoid overwatering"
            ]
            week_2_3 = ["Continue monitoring", "Consider adjusting light conditions if needed"]
            monitoring = "Monitor daily for changes. Watch for spreading of any issues."
        elif severity == "moderate":
            immediate = [
                "Isolate plant if possible to prevent spread",
                "Remove all affected leaves immediately"
            ]
            week_1 = [
                "Continue removing affected leaves as they appear",
                "Reduce watering and ensure good drainage",
                "Increase air circulation around the plant",
                "Consider treatment with appropriate organic solutions"
            ]
            week_2_3 = [
                "Monitor recovery progress",
                "Maintain optimal growing conditions",
                "Continue treatment as needed"
            ]
            monitoring = "Monitor twice daily. Document changes. Seek professional help if condition worsens."
        else:  # high severity
            immediate = [
                "Isolate plant immediately to prevent spread to others",
                "Remove all severely affected parts",
                "Disinfect tools after use"
            ]
            week_1 = [
                "Continue aggressive removal of affected areas",
                "Treat with appropriate solutions based on symptoms",
                "Ensure optimal growing conditions",
                "Consider if plant can be saved or should be discarded"
            ]
            week_2_3 = [
                "Assess if plant is recovering or declining",
                "Continue treatment protocol",
                "Make decision on long-term viability"
            ]
            monitoring = "Monitor multiple times daily. Take photos to track progress. Consider consulting a plant specialist."
        
        return {
            'immediate': immediate,
            'week_1': week_1,
            'week_2_3': week_2_3,
            'monitoring': monitoring
        }


# Global instance
_diagnosis_engine = None

def get_diagnosis_engine() -> DiagnosisEngine:
    """Get or create global diagnosis engine instance."""
    global _diagnosis_engine
    if _diagnosis_engine is None:
        _diagnosis_engine = DiagnosisEngine()
    return _diagnosis_engine

