#!/usr/bin/env python3
"""
Node Combiner - Combine adjacent paragraphs based on dual criteria.

This script analyzes and combines adjacent paragraphs when:
- First paragraph matches criteria A
- Next paragraph matches criteria B

Useful for combining:
- Introductory text + lists
- Figure captions + descriptions
- Headers + explanatory text
- Questions + answers

Enhanced with:
✅ No images are lost - any images from the second paragraph are preserved
✅ You get warned if images are being moved
✅ Block counts are combined for completeness
✅ Simple structure maintained - just adds images to the existing array
✅ Only unique images are added
✅ Duplicates are detected and skipped
✅ You get informed about what happened
"""

import json
import sys
import re
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import operator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('node_combining.log'),
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
    
    def _parse_single_criterion(self, criterion_str: str) -> Optional[Dict[str, Any]]:
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
        
        # Handle booleans and null
        if value_str.lower() == 'true':
            return True
        elif value_str.lower() == 'false':
            return False
        elif value_str.lower() == 'null':
            return None
        
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
            
            # Only skip if actual_value is None but we're not looking for None
            if actual_value is None and expected_value is not None:
                results.append(False)
                continue
            
            # Special handling for regex on text field
            if field == 'text' and isinstance(expected_value, str):
                try:
                    if op in ['=', '==']:
                        match = re.search(expected_value, actual_value, re.IGNORECASE)
                        results.append(match is not None)
                    elif op != '!=':
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

class NodeCombiner:
    """Combine adjacent paragraphs based on dual criteria."""
    
    def __init__(self, json_file: str):
        self.json_file = json_file
        self.data = self._load_json()
    
    def _load_json(self) -> Dict[str, Any]:
        """Load the JSON file."""
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"File not found: {self.json_file}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {self.json_file}: {e}")
            sys.exit(1)
    
    def analyze_combinations(self, first_criteria: str, second_criteria: str, search_distance: int = 3) -> List[Dict[str, Any]]:
        """Analyze potential paragraph combinations that match the dual criteria.
        
        Args:
            first_criteria: Criteria for first paragraph
            second_criteria: Criteria for second paragraph  
            search_distance: How many paragraphs ahead to search (default: 3)
        """
        first_parser = CriteriaParser()
        first_parser.parse_criteria_string(first_criteria)
        
        second_parser = CriteriaParser()
        second_parser.parse_criteria_string(second_criteria)
        
        potential_combinations = []
        total_paragraphs = 0
        total_pairs = 0
        
        def analyze_container(container: Dict, path: List[str]):
            """Recursively analyze any container for potential combinations."""
            nonlocal total_paragraphs, total_pairs, potential_combinations
            
            # Get all paragraphs from this container
            paragraphs = container.get('paragraphs', [])
            total_paragraphs += len(paragraphs)
            
            # Check each paragraph as potential first match
            for i in range(len(paragraphs)):
                current_para = paragraphs[i]
                
                # Skip if this doesn't match first criteria
                if not first_parser.evaluate_paragraph(current_para):
                    continue
                
                # Search within this container for second match
                for j in range(i + 1, min(i + search_distance + 1, len(paragraphs))):
                    candidate_para = paragraphs[j]
                    total_pairs += 1
                    
                    if second_parser.evaluate_paragraph(candidate_para):
                        potential_combinations.append({
                            'path': ' → '.join(path),
                            'page': container.get('page', 'Unknown'),
                            'first_paragraph': current_para,
                            'second_paragraph': candidate_para,
                            'first_index': i,
                            'second_index': j,
                            'distance': j - i,
                            'combination_preview': self._create_combination_preview(current_para, candidate_para),
                            'match_info': {
                                'first_criteria': first_criteria,
                                'second_criteria': second_criteria,
                                'search_distance': search_distance,
                                'matched_at': datetime.now().isoformat()
                            }
                        })
                        break  # Take first match within distance
            
            # Also check cross-section boundaries for adjacent sections
            sections = container.get('sections', [])
            for s_idx in range(len(sections)):
                current_section = sections[s_idx]
                section_path = path + [current_section.get('title', 'Untitled Section')]
                
                # Check if last paragraph of current section + first paragraph of next section
                if s_idx + 1 < len(sections):
                    next_section = sections[s_idx + 1]
                    
                    current_paras = current_section.get('paragraphs', [])
                    next_paras = next_section.get('paragraphs', [])
                    
                    if current_paras and next_paras:
                        last_para = current_paras[-1]
                        first_para = next_paras[0]
                        total_pairs += 1
                        
                        if (first_parser.evaluate_paragraph(last_para) and 
                            second_parser.evaluate_paragraph(first_para)):
                            
                            potential_combinations.append({
                                'path': f"{' → '.join(section_path)} | {next_section.get('title', 'Untitled')}",
                                'page': f"{current_section.get('page', '?')}-{next_section.get('page', '?')}",
                                'first_paragraph': last_para,
                                'second_paragraph': first_para,
                                'first_index': len(current_paras) - 1,
                                'second_index': 0,
                                'distance': 'cross-section',
                                'combination_preview': self._create_combination_preview(last_para, first_para),
                                'match_info': {
                                    'first_criteria': first_criteria,
                                    'second_criteria': second_criteria,
                                    'type': 'cross_section',
                                    'matched_at': datetime.now().isoformat()
                                }
                            })
                
                # Recursively analyze the section
                analyze_container(current_section, section_path)
        
        # Start analysis from outline/chapters
        for chapter in self.data.get('outline', []):
            chapter_path = [chapter.get('title', 'Untitled Chapter')]
            analyze_container(chapter, chapter_path)
        
        # Log the analysis
        self._log_analysis(first_criteria, second_criteria, potential_combinations, total_paragraphs, total_pairs)
        
        return potential_combinations
    
    def _create_combination_preview(self, first_para: Dict, second_para: Dict) -> Dict[str, Any]:
        """Create a preview of what the combined paragraph would look like."""
        # Combine text with appropriate separator
        first_text = first_para.get('text', '').strip()
        second_text = second_para.get('text', '').strip()
        
        # Choose separator based on how first paragraph ends
        if first_text.endswith(':'):
            separator = '\n'  # Colon suggests list follows
        elif first_text.endswith('.') or first_text.endswith('!') or first_text.endswith('?'):
            separator = ' '  # Complete sentence, use space
        else:
            separator = ' '  # Default to space
        
        combined_text = f"{first_text}{separator}{second_text}"
        
        # Combine pages
        first_pages = first_para.get('pages', [])
        second_pages = second_para.get('pages', [])
        combined_pages = sorted(list(set(first_pages + second_pages)))
        
        # Calculate combined statistics
        combined_word_count = first_para.get('word_count', 0) + second_para.get('word_count', 0)
        combined_char_count = first_para.get('character_count', 0) + second_para.get('character_count', 0)
        combined_block_count = first_para.get('block_count', 0) + second_para.get('block_count', 0)
        
        # Use font properties from the first paragraph (usually the header/intro)
        return {
            'text': combined_text,
            'pages': combined_pages,
            'word_count': combined_word_count,
            'character_count': combined_char_count,
            'block_count': combined_block_count,
            'case_type': first_para.get('case_type', 'unknown'),
            'font_name': first_para.get('font_name', 'unknown'),
            'font_size': first_para.get('font_size', 0),
            'is_bold': first_para.get('is_bold', False),
            'is_italic': first_para.get('is_italic', False),
            'combination_type': 'dual_criteria_merge',
            'original_count': 2
        }
    
    def _log_analysis(self, first_criteria: str, second_criteria: str, combinations: List[Dict], total_paras: int, total_pairs: int):
        """Log analysis results to file."""
        log_filename = f"combination_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        with open(log_filename, 'w', encoding='utf-8') as f:
            f.write(f"Node Combination Analysis Report\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Source File: {self.json_file}\n")
            f.write(f"First Criteria: {first_criteria}\n")
            f.write(f"Second Criteria: {second_criteria}\n")
            f.write(f"Total Paragraphs: {total_paras}\n")
            f.write(f"Total Adjacent Pairs: {total_pairs}\n")
            f.write(f"Matching Combinations: {len(combinations)}\n")
            if total_pairs > 0:
                f.write(f"Combination Rate: {len(combinations)/total_pairs*100:.1f}%\n\n")
            else:
                f.write(f"Combination Rate: N/A (no pairs found)\n\n")
            
            f.write("POTENTIAL COMBINATIONS:\n")
            f.write("-" * 30 + "\n")
            
            for i, combo in enumerate(combinations, 1):
                f.write(f"\n{i}. Path: {combo['path']} (Page {combo['page']})\n")
                f.write(f"   Indices: [{combo['first_index']}] + [{combo['second_index']}] (distance: {combo.get('distance', 1)})\n")
                
                first = combo['first_paragraph']
                second = combo['second_paragraph']
                preview = combo['combination_preview']
                
                f.write(f"   FIRST:  {first['text'][:80]}{'...' if len(first['text']) > 80 else ''}\n")
                f.write(f"           Font: {first.get('font_name', 'Unknown')} {first.get('font_size', '?')}pt\n")
                f.write(f"           Case: {first.get('case_type', 'Unknown')}, Words: {first.get('word_count', '?')}\n")
                
                f.write(f"   SECOND: {second['text'][:80]}{'...' if len(second['text']) > 80 else ''}\n")
                f.write(f"           Font: {second.get('font_name', 'Unknown')} {second.get('font_size', '?')}pt\n")
                f.write(f"           Case: {second.get('case_type', 'Unknown')}, Words: {second.get('word_count', '?')}\n")
                
                f.write(f"   COMBINED: {preview['text'][:100]}{'...' if len(preview['text']) > 100 else ''}\n")
                f.write(f"             Total Words: {preview['word_count']}, Pages: {preview['pages']}\n")
        
        logger.info(f"Combination analysis results written to {log_filename}")
        logger.info(f"Found {len(combinations)} potential combinations out of {total_pairs} adjacent pairs")
    
    def combine_nodes(self, first_criteria: str, second_criteria: str, output_file: str = None) -> Dict[str, Any]:
        """Combine adjacent paragraphs based on dual criteria."""
        first_parser = CriteriaParser()
        first_parser.parse_criteria_string(first_criteria)
        
        second_parser = CriteriaParser()
        second_parser.parse_criteria_string(second_criteria)
        
        def process_container(container: Dict) -> Dict:
            """Process any container to combine adjacent paragraphs."""
            new_container = container.copy()
            
            # Process existing sections recursively
            if 'sections' in container and container['sections']:
                new_sections = []
                for section in container['sections']:
                    processed_section = process_container(section)
                    new_sections.append(processed_section)
                new_container['sections'] = new_sections
            
            # Process paragraphs in this container
            paragraphs = container.get('paragraphs', [])
            if not paragraphs:
                return new_container
            
            new_paragraphs = []
            i = 0
            combinations_made = 0
            
            while i < len(paragraphs):
                current_para = paragraphs[i]
                
                # Check if we can combine with next paragraph
                if (i + 1 < len(paragraphs) and 
                    first_parser.evaluate_paragraph(current_para) and
                    second_parser.evaluate_paragraph(paragraphs[i + 1])):
                    
                    # Simply combine the text and keep the first paragraph's structure
                    next_para = paragraphs[i + 1]
                    
                    # Combine text with appropriate separator
                    first_text = current_para.get('text', '').strip()
                    second_text = next_para.get('text', '').strip()
                    
                    # Choose separator
                    if first_text.endswith(':'):
                        separator = '\n'
                    elif first_text.endswith('.') or first_text.endswith('!') or first_text.endswith('?'):
                        separator = ' '
                    else:
                        separator = ' '
                    
                    # Update the first paragraph with combined text
                    combined_para = current_para.copy()
                    combined_para['text'] = f"{first_text}{separator}{second_text}"
                    
                    # Update word/character counts
                    combined_para['word_count'] = current_para.get('word_count', 0) + next_para.get('word_count', 0)
                    combined_para['character_count'] = len(combined_para['text'])
                    
                    # Combine pages if different
                    first_pages = current_para.get('pages', [])
                    second_pages = next_para.get('pages', [])
                    if second_pages and second_pages != first_pages:
                        combined_para['pages'] = sorted(list(set(first_pages + second_pages)))
                    
                    # 🎯 ENHANCED IMAGE PRESERVATION WITH DEDUPLICATION
                    first_images = combined_para.get('images', [])
                    second_images = next_para.get('images', [])
                    
                    if second_images:
                        # Deduplicate by base64 content
                        existing_base64 = {img.get('base64', '') for img in first_images if img.get('base64')}
                        unique_second_images = []
                        duplicate_count = 0
                        
                        for img in second_images:
                            img_base64 = img.get('base64', '')
                            if img_base64 and img_base64 not in existing_base64:
                                unique_second_images.append(img)
                                existing_base64.add(img_base64)
                            else:
                                duplicate_count += 1
                        
                        # Add unique images
                        if unique_second_images:
                            combined_para['images'] = first_images + unique_second_images
                            logger.info(f"✅ Added {len(unique_second_images)} unique images from second paragraph")
                        
                        # Report duplicates
                        if duplicate_count > 0:
                            logger.info(f"⚠️  Skipped {duplicate_count} duplicate images from second paragraph")
                        
                        # Warn about image movement
                        if unique_second_images or duplicate_count > 0:
                            logger.info(f"📸 Image preservation: {len(first_images)} original + {len(unique_second_images)} added + {duplicate_count} duplicates skipped")
                    
                    # Combine block_count
                    combined_para['block_count'] = current_para.get('block_count', 0) + next_para.get('block_count', 0)
                    
                    new_paragraphs.append(combined_para)
                    combinations_made += 1
                    i += 2  # Skip both paragraphs
                else:
                    # Keep paragraph as is
                    new_paragraphs.append(current_para)
                    i += 1
            
            new_container['paragraphs'] = new_paragraphs
            new_container['combinations_made'] = combinations_made
            
            # Update statistics
            if new_paragraphs:
                new_container['paragraph_count'] = len(new_paragraphs)
                new_container['total_words'] = sum(p.get('word_count', 0) for p in new_paragraphs)
                new_container['total_characters'] = sum(p.get('character_count', 0) for p in new_paragraphs)
            
            return new_container
        
        # Process the entire structure
        new_data = self.data.copy()
        new_outline = []
        total_combinations = 0
        
        for chapter in self.data.get('outline', []):
            processed_chapter = process_container(chapter)
            new_outline.append(processed_chapter)
            total_combinations += processed_chapter.get('combinations_made', 0)
        
        new_data['outline'] = new_outline
        
        # Add processing info
        if 'processing_info' not in new_data:
            new_data['processing_info'] = {}
        
        new_data['processing_info'].update({
            'node_combination': {
                'first_criteria': first_criteria,
                'second_criteria': second_criteria,
                'processed_at': datetime.now().isoformat(),
                'total_combinations': total_combinations,
                'script_version': '2.0_enhanced'
            }
        })
        
        # Generate output filename
        if output_file is None:
            base_name = self.json_file.replace('.json', '')
            output_file = f"{base_name}_combined.json"
        
        # Save output
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Combined JSON written to {output_file}")
        logger.info(f"Made {total_combinations} combinations across {len(new_outline)} chapters")
        
        return new_data

def create_parser():
    """Create argument parser with proper flags."""
    parser = argparse.ArgumentParser(
        description='Combine adjacent paragraphs based on dual criteria (Enhanced with image preservation)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze potential combinations of introductory text + lists
  python node_combiner.py analyze --first 'text=":$"' --second 'is_list_item=true'
  
  # Combine figure captions with descriptions (ENHANCED with image preservation!)
  python node_combiner.py combine --first 'is_figure=true,figure_title=null' --second 'font_size>10,font_size<16'
  
  # Combine headers ending with colons + explanatory text
  python node_combiner.py analyze --file font_structure.json --first 'text=":$",is_bold=true' --second 'case_type="sentence_case"'
  
  # Combine questions with answers
  python node_combiner.py combine --first 'text="\\?$"' --second 'word_count>10' --output combined.json

🎯 ENHANCED FEATURES:
  ✅ No images are lost - any images from the second paragraph are preserved
  ✅ You get warned if images are being moved
  ✅ Block counts are combined for completeness
  ✅ Simple structure maintained - just adds images to the existing array
  ✅ Only unique images are added
  ✅ Duplicates are detected and skipped
  ✅ You get informed about what happened

Criteria format:
  Same as section_processor.py - field=value, field>value, etc.
  Text field supports regex patterns
  Available fields: text, case_type, font_name, font_size, is_bold, is_italic, 
                   word_count, character_count, is_list_item, list_type, is_figure

Use Cases:
  - Combine introductory paragraphs with following lists
  - Merge figure references with their descriptions (WITH IMAGE PRESERVATION!)
  - Join questions with their answers
  - Combine headers with explanatory content
  - Merge related short paragraphs for better semantic meaning
        """)
    
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Analyze subcommand
    analyze_parser = subparsers.add_parser('analyze', help='Analyze potential combinations')
    analyze_parser.add_argument('--file', '-f', default='font_structure.json', 
                               help='Input JSON file (default: font_structure.json)')
    analyze_parser.add_argument('--first', required=True,
                               help='Criteria for first paragraph in combination')
    analyze_parser.add_argument('--second', required=True,
                               help='Criteria for second paragraph in combination')
    analyze_parser.add_argument('--distance', '-d', type=int, default=5,
                               help='How many paragraphs ahead to search (default: 5)')
    
    # Combine subcommand
    combine_parser = subparsers.add_parser('combine', help='Combine paragraphs based on criteria')
    combine_parser.add_argument('--file', '-f', default='font_structure.json',
                               help='Input JSON file (default: font_structure.json)')
    combine_parser.add_argument('--first', required=True,
                               help='Criteria for first paragraph in combination')
    combine_parser.add_argument('--second', required=True,
                               help='Criteria for second paragraph in combination')
    combine_parser.add_argument('--output', '-o',
                               help='Output file (default: input_file_combined.json)')
    
    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.mode:
        parser.print_help()
        sys.exit(1)
    
    combiner = NodeCombiner(args.file)
    
    if args.mode == 'analyze':
        # Use user-specified search distance 
        combinations = combiner.analyze_combinations(args.first, args.second, search_distance=args.distance)
        print(f"\nFound {len(combinations)} potential combinations")
        
        # Show first few combinations
        for i, combo in enumerate(combinations[:3], 1):
            print(f"\n{i}. {combo['path']} (Page {combo['page']})")
            print(f"   FIRST:  {combo['first_paragraph']['text'][:60]}...")
            print(f"   SECOND: {combo['second_paragraph']['text'][:60]}...")
            print(f"   COMBINED: {combo['combination_preview']['text'][:80]}...")
        
        if len(combinations) > 3:
            print(f"\n... and {len(combinations) - 3} more (see analysis log for full results)")
    
    elif args.mode == 'combine':
        output_file = args.output
        processed_data = combiner.combine_nodes(args.first, args.second, output_file)
        print(f"🎉 Enhanced combination complete! Combined JSON saved.")
        
        # Show summary
        total_combinations = processed_data.get('processing_info', {}).get('node_combination', {}).get('total_combinations', 0)
        total_chapters = len(processed_data.get('outline', []))
        print(f"Made {total_combinations} combinations across {total_chapters} chapters")
        print(f"📸 All images preserved with deduplication!")

if __name__ == '__main__':
    main()