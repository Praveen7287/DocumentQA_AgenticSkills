#!/usr/bin/env python3
"""
Skill Manager for Agent Skills Integration
Dynamically loads and manages skills from SKILL.md files
"""

import yaml
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import ollama

@dataclass
class SkillMetadata:
    """Skill metadata from YAML frontmatter"""
    name: str
    description: str
    license: Optional[str] = None
    compatibility: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None
    allowed_tools: Optional[List[str]] = None
    path: Optional[Path] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return asdict(self)

class SkillManager:
    """Manages discovery, loading, and execution of Agent Skills"""
    
    def __init__(self, skills_dir: str = "skills"):
        self.skills_dir = Path(skills_dir)
        self.skills: Dict[str, Dict[str, Any]] = {}
        self.ollama_model = "qwen3:1.7b"  # Default model
        
        # Ensure skills directory exists
        if not self.skills_dir.exists():
            self.skills_dir.mkdir(parents=True)
        
        self.discover_skills()
    
    def discover_skills(self) -> None:
        """Discover all skills in the skills directory"""
        print(f"🔍 Discovering skills in {self.skills_dir}...")
        
        for skill_dir in self.skills_dir.iterdir():
            if skill_dir.is_dir():
                skill_file = skill_dir / "SKILL.md"
                if skill_file.exists():
                    try:
                        skill_data = self.load_skill(skill_file)
                        if skill_data:
                            self.skills[skill_data['metadata'].name] = skill_data
                            print(f"  ✓ Loaded skill: {skill_data['metadata'].name}")
                    except Exception as e:
                        print(f"  ✗ Error loading skill {skill_dir.name}: {e}")
    
    def load_skill(self, skill_file: Path) -> Optional[Dict[str, Any]]:
        """
        Load a skill from SKILL.md file
        
        Args:
            skill_file: Path to SKILL.md file
            
        Returns:
            Dictionary containing skill metadata and content
        """
        try:
            content = skill_file.read_text(encoding='utf-8')
            
            # Parse YAML frontmatter
            frontmatter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
            if not frontmatter_match:
                raise ValueError(f"No YAML frontmatter found in {skill_file}")
            
            frontmatter = yaml.safe_load(frontmatter_match.group(1))
            body = content[frontmatter_match.end():].strip()
            
            # Validate required fields
            if 'name' not in frontmatter or 'description' not in frontmatter:
                raise ValueError(f"Missing required fields in {skill_file}")
            
            # Create skill metadata
            metadata = SkillMetadata(
                name=frontmatter['name'],
                description=frontmatter['description'],
                license=frontmatter.get('license'),
                compatibility=frontmatter.get('compatibility'),
                metadata=frontmatter.get('metadata', {}),
                allowed_tools=frontmatter.get('allowed_tools', '').split() if frontmatter.get('allowed_tools') else None,
                path=skill_file.parent
            )
            
            return {
                'metadata': metadata,
                'frontmatter': frontmatter,
                'body': body,
                'full_content': content
            }
            
        except Exception as e:
            print(f"Error parsing {skill_file}: {e}")
            return None
    
    def get_available_skills_xml(self) -> str:
        """
        Generate XML representation of available skills for agent prompts
        
        Returns:
            XML string with skill information
        """
        skills_xml = "<available_skills>\n"
        
        for skill_name, skill_data in self.skills.items():
            metadata = skill_data['metadata']
            skills_xml += f"""  <skill>
    <name>{metadata.name}</name>
    <description>{metadata.description}</description>
    <path>{metadata.path}</path>
  </skill>\n"""
        
        skills_xml += "</available_skills>"
        return skills_xml
    
    def get_skill_instructions(self, skill_name: str) -> Optional[str]:
        """
        Get the full instructions for a specific skill
        
        Args:
            skill_name: Name of the skill
            
        Returns:
            Skill instructions or None if not found
        """
        if skill_name in self.skills:
            return self.skills[skill_name]['full_content']
        return None
    
    def activate_skill(self, skill_name: str, context: Dict[str, Any]) -> str:
        """
        Activate a skill by sending it to Ollama with context
        
        Args:
            skill_name: Name of the skill to activate
            context: Context data for the skill
            
        Returns:
            Response from Ollama
        """
        if skill_name not in self.skills:
            return f"Skill '{skill_name}' not found."
        
        skill_data = self.skills[skill_name]
        
        # Prepare the prompt with skill instructions
        prompt = f"""You are executing the skill: {skill_name}

SKILL INSTRUCTIONS:
{skill_data['body']}

CONTEXT DATA:
{json.dumps(context, indent=2)}

TASK: Based on the skill instructions above, process the context data and provide the result.
        
RESULT:"""
        
        try:
            response = ollama.generate(
                model=self.ollama_model,
                prompt=prompt,
                options={
                    'temperature': 0.3,
                    'num_predict': 1000
                }
            )
            return response['response']
        except Exception as e:
            return f"Error executing skill: {str(e)}"
    
    def match_skill_to_query(self, user_query: str) -> List[str]:
        """
        Match user query to available skills based on descriptions
        
        Args:
            user_query: User's query
            
        Returns:
            List of skill names ordered by relevance
        """
        # Simple keyword matching - can be enhanced with embeddings
        query_lower = user_query.lower()
        matched_skills = []
        
        for skill_name, skill_data in self.skills.items():
            description = skill_data['metadata'].description.lower()
            
            # Check for keywords in description
            keywords = ['document', 'pdf', 'text', 'extract', 'retrieve', 
                       'search', 'query', 'summarize', 'summary', 'answer']
            
            if any(keyword in query_lower and keyword in description 
                  for keyword in keywords):
                matched_skills.append(skill_name)
        
        return matched_skills
    
    def execute_workflow(self, workflow: List[str], initial_context: Dict) -> Dict:
        """
        Execute a sequence of skills as a workflow
        
        Args:
            workflow: List of skill names to execute in order
            initial_context: Initial context data
            
        Returns:
            Final result after workflow execution
        """
        context = initial_context.copy()
        
        for i, skill_name in enumerate(workflow):
            print(f"🚀 Executing step {i+1}/{len(workflow)}: {skill_name}")
            
            result = self.activate_skill(skill_name, context)
            
            # Update context with result
            context[f"step_{i+1}_{skill_name}"] = result
            context['previous_result'] = result
        
        return context

# Helper function to create skill templates
def create_skill_template(skill_name: str, description: str) -> str:
    """Create a template for a new skill"""
    template = f"""---
name: {skill_name}
description: {description}
license: MIT
metadata:
  author: your-name
  version: "1.0"
---

# {skill_name.replace('-', ' ').title()} Instructions

## Purpose
Describe what this skill does and when to use it.

## Step-by-Step Instructions
1. First step
2. Second step
3. Third step

## Examples
Example input and output.

## Common Issues and Solutions
- Issue 1: Solution 1
- Issue 2: Solution 2

## Implementation Notes
Any technical details or requirements.
"""
    return template

if __name__ == "__main__":
    # Test the skill manager
    manager = SkillManager()
    print(f"\n📋 Available skills: {list(manager.skills.keys())}")
    print(f"\n📄 Skills XML:\n{manager.get_available_skills_xml()}")