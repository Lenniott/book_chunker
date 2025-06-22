import json
import sys
import re
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Any, Tuple
import operator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('section_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CriteriaParser:
    """Parse and evaluate filtering criteria from command line."""
    
    OPERATORS = {
        '>=': operator.ge,
        '<=': operator.le,
        '>': operator.gt,
        '<': operator.lt,
        '==': operator.eq,
        '!=': operator.ne,
        '=': operator.eq,  # Allow single = as well
    }
    
    def __init__(self):
        self.criteria = []
        self.logic_ops = []  # 'and' or 'or' between criteria
    
    def parse_criteria_string(self, criteria_str: str):
        """Parse criteria string like 'case_type="all_caps",word_count<10,is_bold=true'"""
        self.criteria = []
        self.logic_ops = []
        
        # Split by comma but respect quoted strings
        parts = self._split_respecting_quotes(criteria_str, ',')
        
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
                
            # Check for OR logic (if part starts with 'or ')
            if part.lower().startswith('or '):
                self.logic_ops.append('or')
                part = part[3:].strip()
            else:
                if i > 0:  # Default to AND for subsequent criteria
                    self.logic_ops.append('and')
            
            criterion = self._parse_single_criterion(part)
            if criterion:
                self.criteria.append(criterion)
    
    def _split_respecting_quotes(self, text: str, delimiter: str) -> List[str]:
        """Split text by delimiter but respect quoted strings."""
        parts = []
        current = ""
        in_quotes = False
        quote_char = None
        
        for char in text:
            if char in ['"', "'"] and not in_quotes:
                in_quotes = True
                quote_char = char
                current += char
            elif char == quote_char and in_quotes:
                in_quotes = False
                quote_char = None
                current += char
            elif char == delimiter and not in_quotes:
                parts.append(current)
                current = ""
            else:
                current += char
        
        if current:
            parts.append(current)
        
        return parts
    
    def _parse_single_criterion(self, criterion_str: str) -> Dict[str, Any]:
        """Parse a single criterion like 'word_count<10' or 'case_type="all_caps"'"""
        # Find the operator
        op_found = None
        op_pos = -1
        
        # Check for two-character operators first
        for op in ['>=', '<=', '==', '!=']:
            pos = criterion_str.find(op)
            if pos != -1:
                op_found = op
                op_pos = pos
                break
        
        # Check for single-character operators
        if op_found is None:
            for op in ['>', '<', '=']:
                pos = criterion_str.find(op)
                if pos != -1:
                    op_found = op
                    op_pos = pos
                    break
        
        if op_found is None:
            logger.warning(f"No valid operator found in criterion: {criterion_str}")
            return None
        
        field = criterion_str[:op_pos].strip()
        value_str = criterion_str[op_pos + len(op_found):].strip()
        
        # Parse the value
        value = self._parse_value(value_str)
        
        return {
            'field': field,
            'operator': op_found,
            'value': value,
            'raw': criterion_str
        }
    
    def _parse_value(self, value_str: str) -> Any:
        """Parse a value string to appropriate Python type."""
        value_str = value_str.strip()
        
        # Handle quoted strings
        if (value_str.startswith('"') and value_str.endswith('"')) or \
           (value_str.startswith("'") and value_str.endswith("'")):
            return value_str[1:-1]  # Remove quotes
        
        # Handle booleans
        if value_str.lower() == 'true':
            return True
        elif value_str.lower() == 'false':
            return False
        
        # Handle numbers
        try:
            if '.' in value_str:
                return float(value_str)
            else:
                return int(value_str)
        except ValueError:
            pass
        
        # Return as string if nothing else works
        return value_str
    
    def evaluate_paragraph(self, paragraph: Dict[str, Any]) -> bool:
        """Evaluate if a paragraph matches all criteria."""
        if not self.criteria:
            return True
        
        results = []
        
        for criterion in self.criteria:
            field = criterion['field']
            op = criterion['operator']
            expected_value = criterion['value']
            
            # Get the actual value from paragraph
            actual_value = paragraph.get(field)
            
            if actual_value is None:
                results.append(False)
                continue
            
            # Special handling for regex on text field
            if field == 'text' and isinstance(expected_value, str):
                try:
                    if op in ['=', '==']:
                        match = re.search(expected_value, actual_value, re.IGNORECASE)
                        results.append(match is not None)
                    elif op == '!=':
                        match = re.search(expected_value, actual_value, re.IGNORECASE)
                        results.append(match is None)
                    else:
                        results.append(False)  # Invalid regex operation
                    continue
                except re.error:
                    logger.warning(f"Invalid regex pattern: {expected_value}")
                    results.append(False)
                    continue
            
            # Standard comparison
            try:
                op_func = self.OPERATORS.get(op)
                if op_func:
                    result = op_func(actual_value, expected_value)
                    results.append(result)
                else:
                    results.append(False)
            except TypeError:
                # Type mismatch in comparison
                results.append(False)
        
        # Apply logic operators
        if not results:
            return True
        
        final_result = results[0]
        for i, logic_op in enumerate(self.logic_ops):
            if i + 1 < len(results):
                if logic_op == 'or':
                    final_result = final_result or results[i + 1]
                else:  # 'and'
                    final_result = final_result and results[i + 1]
        
        return final_result

class SectionProcessor:
    """Main class for processing font-aware JSON files."""
    
    def __init__(self, json_file: str):
        self.json_file = json_file
        self.data = self._load_json()
    
    def _load_json(self) -> Dict[str, Any]:
        """Load the font-aware JSON file."""
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"File not found: {self.json_file}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {self.json_file}: {e}")
            sys.exit(1)
    
    def analyze_paragraphs(self, criteria_str: str) -> List[Dict[str, Any]]:
        """Analyze paragraphs that match the given criteria."""
        parser = CriteriaParser()
        parser.parse_criteria_string(criteria_str)
        
        matching_paragraphs = []
        total_paragraphs = 0
        
        # Search through all chapters and their paragraphs
        for chapter in self.data.get('outline', []):
            chapter_title = chapter.get('title', 'Untitled Chapter')
            
            # Check if this chapter has sections (processed JSON) or direct paragraphs (original JSON)
            if 'sections' in chapter:
                # This is a processed JSON with sections
                for section in chapter.get('sections', []):
                    section_title = section.get('title', 'Untitled Section')
                    
                    # Check header paragraph if it exists
                    if 'header_paragraph' in section:
                        total_paragraphs += 1
                        if parser.evaluate_paragraph(section['header_paragraph']):
                            matching_paragraphs.append({
                                'chapter': chapter_title,
                                'section': section_title,
                                'chapter_page': chapter.get('page', 'Unknown'),
                                'paragraph': section['header_paragraph'],
                                'match_info': {
                                    'criteria': criteria_str,
                                    'matched_at': datetime.now().isoformat()
                                }
                            })
                    
                    # Check paragraphs in the section
                    for paragraph in section.get('paragraphs', []):
                        total_paragraphs += 1
                        
                        if parser.evaluate_paragraph(paragraph):
                            matching_paragraphs.append({
                                'chapter': chapter_title,
                                'section': section_title,
                                'chapter_page': chapter.get('page', 'Unknown'),
                                'paragraph': paragraph,
                                'match_info': {
                                    'criteria': criteria_str,
                                    'matched_at': datetime.now().isoformat()
                                }
                            })
            else:
                # This is an original JSON with direct paragraphs
                for paragraph in chapter.get('paragraphs', []):
                    total_paragraphs += 1
                    
                    if parser.evaluate_paragraph(paragraph):
                        matching_paragraphs.append({
                            'chapter': chapter_title,
                            'chapter_page': chapter.get('page', 'Unknown'),
                            'paragraph': paragraph,
                            'match_info': {
                                'criteria': criteria_str,
                                'matched_at': datetime.now().isoformat()
                            }
                        })
        
        # Log the analysis
        self._log_analysis(criteria_str, matching_paragraphs, total_paragraphs)
        
        return matching_paragraphs
    
    def _log_analysis(self, criteria_str: str, matches: List[Dict], total: int):
        """Log analysis results to file."""
        log_filename = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        with open(log_filename, 'w', encoding='utf-8') as f:
            f.write(f"Section Processing Analysis Report\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Source File: {self.json_file}\n")
            f.write(f"Criteria: {criteria_str}\n")
            f.write(f"Total Paragraphs: {total}\n")
            f.write(f"Matching Paragraphs: {len(matches)}\n")
            if total > 0:
                f.write(f"Match Rate: {len(matches)/total*100:.1f}%\n\n")
            else:
                f.write(f"Match Rate: N/A (no paragraphs found)\n\n")
            
            f.write("MATCHING PARAGRAPHS:\n")
            f.write("-" * 30 + "\n")
            
            for i, match in enumerate(matches, 1):
                f.write(f"\n{i}. Chapter: {match['chapter']} (Page {match['chapter_page']})\n")
                para = match['paragraph']
                f.write(f"   Text: {para['text'][:100]}{'...' if len(para['text']) > 100 else ''}\n")
                f.write(f"   Font: {para.get('font_name', 'Unknown')} {para.get('font_size', 'Unknown')}pt\n")
                f.write(f"   Case: {para.get('case_type', 'Unknown')}\n")
                f.write(f"   Bold: {para.get('is_bold', 'Unknown')}\n")
                f.write(f"   Words: {para.get('word_count', 'Unknown')}\n")
        
        logger.info(f"Analysis results written to {log_filename}")
        logger.info(f"Found {len(matches)} matching paragraphs out of {total} total")
    
    def process_into_sections(self, criteria_str: str, output_file: str = None) -> Dict[str, Any]:
        """Process the JSON to add sections based on criteria."""
        parser = CriteriaParser()
        parser.parse_criteria_string(criteria_str)
        
        # Create new structure
        new_data = self.data.copy()
        new_outline = []
        
        for chapter in self.data.get('outline', []):
            new_chapter = {
                'title': chapter.get('title'),
                'page': chapter.get('page'),
                'paragraph_count': chapter.get('paragraph_count', 0),
                'total_words': chapter.get('total_words', 0),
                'total_characters': chapter.get('total_characters', 0),
                'font_analysis': chapter.get('font_analysis', {}),
                'sections': []
            }
            
            # Copy node_path if it exists
            if 'node_path' in chapter:
                new_chapter['node_path'] = chapter['node_path']
            
            # Check if this chapter already has sections (nested processing)
            if 'sections' in chapter:
                # Process existing sections to create sub-sections
                for existing_section in chapter.get('sections', []):
                    new_section = existing_section.copy()
                    new_section['sections'] = []  # Add sub-sections
                    
                    current_subsection = None
                    subsection_paragraphs = []
                    
                    # Check header paragraph first
                    if 'header_paragraph' in existing_section:
                        if parser.evaluate_paragraph(existing_section['header_paragraph']):
                            # Header becomes a subsection
                            current_subsection = {
                                'title': existing_section['header_paragraph'].get('text', 'Untitled Subsection'),
                                'page': existing_section['header_paragraph'].get('pages', [0])[0] if existing_section['header_paragraph'].get('pages') else 0,
                                'header_paragraph': existing_section['header_paragraph'],
                                'paragraphs': []
                            }
                    
                    # Process paragraphs in the section
                    for paragraph in existing_section.get('paragraphs', []):
                        if parser.evaluate_paragraph(paragraph):
                            # Save previous subsection if it exists
                            if current_subsection is not None:
                                current_subsection['paragraphs'] = subsection_paragraphs
                                current_subsection['paragraph_count'] = len(subsection_paragraphs)
                                current_subsection['total_words'] = sum(p.get('word_count', 0) for p in subsection_paragraphs)
                                current_subsection['total_characters'] = sum(p.get('character_count', 0) for p in subsection_paragraphs)
                                new_section['sections'].append(current_subsection)
                            
                            # Start new subsection
                            current_subsection = {
                                'title': paragraph.get('text', 'Untitled Subsection'),
                                'page': paragraph.get('pages', [0])[0] if paragraph.get('pages') else 0,
                                'header_paragraph': paragraph,
                                'paragraphs': []
                            }
                            subsection_paragraphs = []
                        else:
                            # Add to current subsection
                            subsection_paragraphs.append(paragraph)
                    
                    # Handle remaining paragraphs
                    if current_subsection is not None:
                        current_subsection['paragraphs'] = subsection_paragraphs
                        current_subsection['paragraph_count'] = len(subsection_paragraphs)
                        current_subsection['total_words'] = sum(p.get('word_count', 0) for p in subsection_paragraphs)
                        current_subsection['total_characters'] = sum(p.get('character_count', 0) for p in subsection_paragraphs)
                        new_section['sections'].append(current_subsection)
                    elif subsection_paragraphs:
                        # No subsections found, keep paragraphs at section level
                        new_section['paragraphs'] = subsection_paragraphs
                    
                    # Update section statistics
                    new_section['section_count'] = len(new_section.get('sections', []))
                    new_chapter['sections'].append(new_section)
                    
            else:
                # Original processing for chapters with direct paragraphs
                current_section = None
                section_paragraphs = []
                
                for paragraph in chapter.get('paragraphs', []):
                    # Check if this paragraph should start a new section
                    if parser.evaluate_paragraph(paragraph):
                        # Save previous section if it exists
                        if current_section is not None:
                            current_section['paragraphs'] = section_paragraphs
                            current_section['paragraph_count'] = len(section_paragraphs)
                            current_section['total_words'] = sum(p.get('word_count', 0) for p in section_paragraphs)
                            current_section['total_characters'] = sum(p.get('character_count', 0) for p in section_paragraphs)
                            new_chapter['sections'].append(current_section)
                        
                        # Start new section with this paragraph as the title/header
                        current_section = {
                            'title': paragraph.get('text', 'Untitled Section'),
                            'page': paragraph.get('pages', [0])[0] if paragraph.get('pages') else 0,
                            'header_paragraph': paragraph,  # Store the header paragraph
                            'paragraphs': []
                        }
                        section_paragraphs = []
                    else:
                        # Add to current section
                        section_paragraphs.append(paragraph)
                
                # Handle remaining paragraphs
                if current_section is not None:
                    current_section['paragraphs'] = section_paragraphs
                    current_section['paragraph_count'] = len(section_paragraphs)
                    current_section['total_words'] = sum(p.get('word_count', 0) for p in section_paragraphs)
                    current_section['total_characters'] = sum(p.get('character_count', 0) for p in section_paragraphs)
                    new_chapter['sections'].append(current_section)
                elif section_paragraphs:
                    # No sections found, create a default section
                    default_section = {
                        'title': 'Content',
                        'page': chapter.get('page', 0),
                        'paragraphs': section_paragraphs,
                        'paragraph_count': len(section_paragraphs),
                        'total_words': sum(p.get('word_count', 0) for p in section_paragraphs),
                        'total_characters': sum(p.get('character_count', 0) for p in section_paragraphs)
                    }
                    new_chapter['sections'].append(default_section)
            
            # Update chapter statistics
            new_chapter['section_count'] = len(new_chapter['sections'])
            
            new_outline.append(new_chapter)
        
        new_data['outline'] = new_outline
        
        # Add processing info
        if 'processing_info' not in new_data:
            new_data['processing_info'] = {}
        
        new_data['processing_info'].update({
            'section_processing': {
                'criteria': criteria_str,
                'processed_at': datetime.now().isoformat(),
                'script_version': '1.0'
            }
        })
        
        # Save to file
        if output_file is None:
            output_file = self.json_file.replace('.json', '_sectioned.json')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Processed JSON with sections written to {output_file}")
        
        # Log processing summary
        total_sections = sum(chapter.get('section_count', 0) for chapter in new_outline)
        logger.info(f"Created {total_sections} sections across {len(new_outline)} chapters")
        
        return new_data

def create_parser():
    """Create argument parser with proper flags."""
    parser = argparse.ArgumentParser(
        description='Process font-aware JSON files to analyze and create sections',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python section_processor.py analyze --criteria 'case_type="all_caps"'
  python section_processor.py analyze --file font_structure.json --criteria 'case_type="all_caps",word_count<10'
  python section_processor.py process --criteria 'case_type="all_caps",is_bold=true' --output sectioned.json
  python section_processor.py analyze --criteria 'text="chapter",font_size>14.0'

Criteria format:
  field=value, field>value, field<value, field>=value, field<=value, field!=value
  Use 'or' between criteria for OR logic (default is AND)
  Text field supports regex patterns
  Available fields: text, case_type, font_name, font_size, is_bold, is_italic, word_count, character_count
        """)
    
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Analyze subcommand
    analyze_parser = subparsers.add_parser('analyze', help='Analyze paragraphs matching criteria')
    analyze_parser.add_argument('--file', '-f', default='font_structure.json', 
                               help='Input JSON file (default: font_structure.json)')
    analyze_parser.add_argument('--criteria', '-c', required=True,
                               help='Filtering criteria string')
    
    # Process subcommand
    process_parser = subparsers.add_parser('process', help='Process JSON to create sections')
    process_parser.add_argument('--file', '-f', default='font_structure.json',
                               help='Input JSON file (default: font_structure.json)')
    process_parser.add_argument('--criteria', '-c', required=True,
                               help='Criteria for section headers')
    process_parser.add_argument('--output', '-o',
                               help='Output file (default: input_file_sectioned.json)')
    
    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.mode:
        parser.print_help()
        sys.exit(1)
    
    processor = SectionProcessor(args.file)
    
    if args.mode == 'analyze':
        matches = processor.analyze_paragraphs(args.criteria)
        print(f"\nFound {len(matches)} matching paragraphs")
        
        # Show first few matches
        for i, match in enumerate(matches[:5], 1):
            para = match['paragraph']
            location = match['chapter']
            if 'section' in match:
                location += f" → {match['section']}"
            location += f" (Page {match['chapter_page']})"
            
            print(f"\n{i}. {location}")
            print(f"   Text: {para['text'][:80]}{'...' if len(para['text']) > 80 else ''}")
            print(f"   Font: {para.get('font_name', 'Unknown')} {para.get('font_size', 'Unknown')}pt, Bold: {para.get('is_bold', 'Unknown')}")
            print(f"   Case: {para.get('case_type', 'Unknown')}")
        
        if len(matches) > 5:
            print(f"\n... and {len(matches) - 5} more (see analysis log for full results)")
    
    elif args.mode == 'process':
        output_file = args.output
        processed_data = processor.process_into_sections(args.criteria, output_file)
        print(f"Processing complete! Structured JSON saved.")
        
        # Show summary
        total_sections = sum(chapter.get('section_count', 0) for chapter in processed_data.get('outline', []))
        total_chapters = len(processed_data.get('outline', []))
        print(f"Created {total_sections} sections across {total_chapters} chapters")

if __name__ == '__main__':
    main() 